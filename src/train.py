import datetime
import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from datasets import Dataset, DatasetDict
from fire import Fire
from loguru import logger
from peft import get_peft_model, AdaLoraConfig, PeftModel, PeftConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, DefaultDataCollator, AutoModel

from fixtures import AVAILABLE_CLS_LOSSES, AVAILABLE_SSL_LOSSES

DATA_PATH = 'data/'
ROOT_SAVE_PATH = 'output/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'full_dataset.csv')
STAGES = ['train', 'val', 'test']


def _get_peft_model(target_r, init_r, lora_alpha, lora_dropout, model_name: str) -> nn.Module:
    base_model = EmbeddingModel(model_name)

    peft_config = AdaLoraConfig(
        target_r=target_r,
        init_r=init_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        target_modules=['query', 'value'],
    )
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.model = get_peft_model(base_model.model, peft_config)
    trainable_params, all_param = base_model.model.get_nb_trainable_parameters()
    logger.info('Model created')
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} ||"
        f" trainable%: {100 * trainable_params / all_param}"
    )
    return base_model


def _prepare_train_test_split(max_classes):
    raw_df = pd.read_csv(RAW_DATA_PATH).dropna()
    unique_labels = raw_df['author_id'].unique()
    if max_classes is not None and len(unique_labels) > max_classes:
        # Randomly sample the unique labels
        sampled_labels = sorted(unique_labels)[:max_classes]
        raw_df = raw_df[raw_df['author_id'].isin(sampled_labels)]
        label2idx = {label: i for i, label in enumerate(sampled_labels)}
        raw_df['author_id'] = raw_df['author_id'].map(lambda x: label2idx[x])

    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=0)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=0)
    for df, stage in zip([train_df, val_df, test_df], STAGES):
        df.reset_index(drop=True).to_csv(os.path.join(DATA_PATH, f'{stage}.csv'))


def _setup_save_dir(save_dir):
    if save_dir is None:
        now = datetime.datetime.now()
        save_dir = os.path.join(ROOT_SAVE_PATH,
                                f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def _compose_dataset(tokenizer, model_max_tokens, max_classes=None):
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['text'], max_length=model_max_tokens, padding='max_length', truncation=True)
        model_inputs['labels'] = examples['labels']
        return model_inputs

    if not all([os.path.exists(os.path.join(DATA_PATH, f'{stage}.csv')) for stage in STAGES]):
        _prepare_train_test_split(max_classes)

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv')).dropna()
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val.csv')).dropna()
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv')).dropna()
    logger.info(f'Train data len: {len(train_df)} | Validation data len: {len(val_df)} | Test data len: {len(test_df)}')
    for stage in STAGES:
        os.remove(os.path.join(DATA_PATH, f'{stage}.csv'))

    train_text, train_labels = train_df['text'].to_list(), train_df['author_id'].to_list()
    val_text, val_labels = val_df['text'].to_list(), val_df['author_id'].to_list()
    test_text, test_labels = test_df['text'].to_list(), test_df['author_id'].to_list(),

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict({"text": train_text, "labels": train_labels})
    dataset['val'] = Dataset.from_dict({"text": val_text, "labels": val_labels})
    dataset['test'] = Dataset.from_dict({"text": test_text, "labels": test_labels})

    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc='Tokenizing data', remove_columns=['text'])
    return tokenized_dataset


def _compose_apn_dataset(tokenizer, model_max_tokens, num_triplets=None):
    def preprocess_function(examples):
        model_inputs = dict()
        anchors = tokenizer(examples['anchor'], max_length=model_max_tokens, padding='max_length', truncation=True)
        positives = tokenizer(examples['positive'], max_length=model_max_tokens, padding='max_length', truncation=True)
        negatives = tokenizer(examples['negative'], max_length=model_max_tokens, padding='max_length', truncation=True)

        model_inputs['anchor_input_ids'] = anchors['input_ids']
        model_inputs['anchor_attention_mask'] = anchors['attention_mask']

        model_inputs['positive_input_ids'] = positives['input_ids']
        model_inputs['positive_attention_mask'] = positives['attention_mask']

        model_inputs['negative_input_ids'] = negatives['input_ids']
        model_inputs['negative_attention_mask'] = negatives['attention_mask']

        return model_inputs

    df = pd.read_csv(RAW_DATA_PATH).dropna()
    if num_triplets is None:
        num_triplets = len(df)
    elif type(num_triplets) is float:
        num_triplets = int(len(df) * num_triplets)

    unique_author_ids = df['author_id'].unique()
    # Pre-filter the DataFrame to get the indices for each author
    author_indices = {author_id: df[df['author_id'] == author_id].index.tolist() for author_id in unique_author_ids}

    anchor = []
    positive = []
    negative = []
    for _ in tqdm(range(num_triplets), desc='Generating triplets'):
        # Randomly select an author_id
        random_author_id = np.random.choice(unique_author_ids)
        random_author_indices = author_indices[random_author_id]

        # Select the anchor text
        anchor_idx = random_author_indices[np.random.randint(len(random_author_indices))]
        anchor.append(df.loc[anchor_idx, 'text'])

        # Select the positive text (different from anchor)
        positive_idx = random_author_indices[np.random.randint(len(random_author_indices))]
        while positive_idx == anchor_idx:
            positive_idx = random_author_indices[np.random.randint(len(random_author_indices))]
        positive.append(df.loc[positive_idx, 'text'])

        # Select the negative text (from a random different author)
        random_negative_author_id = np.random.choice([idx for idx in unique_author_ids if idx != random_author_id])
        random_negative_author_indices = author_indices[random_negative_author_id]
        negative_idx = random_negative_author_indices[np.random.randint(len(random_negative_author_indices))]
        negative.append(df.loc[negative_idx, 'text'])

    train_anchor, val_anchor, train_positive, val_positive, train_negative, val_negative = \
        train_test_split(anchor, positive, negative, test_size=0.2, random_state=0)
    val_anchor, test_anchor, val_positive, test_positive, val_negative, test_negative = \
        train_test_split(val_anchor, val_positive, val_negative, test_size=0.5, random_state=0)

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict(
        {"anchor": train_anchor, "positive": train_positive, "negative": train_negative})
    dataset['test'] = Dataset.from_dict({"anchor": test_anchor, "positive": test_positive, "negative": test_negative})
    dataset['val'] = Dataset.from_dict({"anchor": val_anchor, "positive": val_positive, "negative": val_negative})

    tokenized_dataset = dataset.map(preprocess_function,
                                    batched=True,
                                    desc='Tokenizing dataset',
                                    remove_columns=['anchor', 'positive', 'negative'])
    return tokenized_dataset


def train_classifier(model,
                     tokenized_dataset,
                     loss_fn,
                     tokenizer,
                     epochs,
                     save_dir,
                     train_batch_size,
                     eval_batch_size,
                     learning_rate,
                     gradient_accumulation_steps,
                     weight_decay,
                     ):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = next(model.parameters()).device
    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=train_batch_size,
                                  collate_fn=data_collator, drop_last=True, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(tokenized_dataset['val'], batch_size=eval_batch_size,
                                collate_fn=data_collator, drop_last=True, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': loss_fn.parameters()}], lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    best_val_loss = float('inf')

    logger.info('Training started')
    # I am aware that huggingface have classes TrainerArguments and Trainer,
    # but they did not connect with peft (for some reason, LoRA drops some columns from the dataset)
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False, desc='Training'):
            batch = batch.to(device)
            labels = batch.pop('labels')
            pred = model(**batch)
            loss = loss_fn(pred, labels)
            loss.backward()
            train_loss += loss.item()
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_loss /= len(train_dataloader)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        val_loss = 0
        correctly_classified = 0
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False, desc='Validating'):
            batch = batch.to(device)
            labels = batch.pop('labels')
            with torch.inference_mode():
                pred = model(**batch)
                loss = loss_fn(pred, labels)
                pred = loss_fn.fc(pred)  # ArcFace has a trainable projection
            correctly_classified += (pred.argmax(dim=-1) == labels).sum().item()
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        accuracy = correctly_classified / (len(val_dataloader) * eval_batch_size)
        logger.info(f'Epoch={epoch + 1}/{epochs} | Train loss = {train_loss:.6f} |'
                    f' Val loss = {val_loss:.6f} | Val accuracy = {accuracy:.6f}')
        # Maybe do the logging over batches?
        wandb.log({"cls_training_loss": train_loss, "cls_validation_loss": val_loss,
                   "cls_validation_accuracy": accuracy}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(str(os.path.join(save_dir, 'peft_encoder_weights')))
            logger.info(f'Best model params saved at {save_dir}')


def train_embeddings(model,
                     tokenized_dataset,
                     loss_fn,
                     epochs,
                     save_dir,
                     train_batch_size,
                     eval_batch_size,
                     learning_rate,
                     weight_decay,
                     device):
    data_collator = DefaultDataCollator()

    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=train_batch_size,
                                  collate_fn=data_collator, drop_last=True, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(tokenized_dataset['val'], batch_size=eval_batch_size,
                                collate_fn=data_collator, drop_last=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader, desc='Training', leave=False):
            anchor = {'input_ids': batch['anchor_input_ids'].to(device),
                      'attention_mask': batch['anchor_attention_mask'].to(device)}
            positive = {'input_ids': batch['positive_input_ids'].to(device),
                        'attention_mask': batch['positive_attention_mask'].to(device)}
            negative = {'input_ids': batch['negative_input_ids'].to(device),
                        'attention_mask': batch['negative_attention_mask'].to(device)}

            anchor_emb = model(**anchor)
            positive_emb = model(**positive)
            negative_emb = model(**negative)

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            anchor = {'input_ids': batch['anchor_input_ids'].to(device),
                      'attention_mask': batch['anchor_attention_mask'].to(device)}
            positive = {'input_ids': batch['positive_input_ids'].to(device),
                        'attention_mask': batch['positive_attention_mask'].to(device)}
            negative = {'input_ids': batch['negative_input_ids'].to(device),
                        'attention_mask': batch['negative_attention_mask'].to(device)}

            with torch.no_grad():
                anchor_emb = model(**anchor)
                positive_emb = model(**positive)
                negative_emb = model(**negative)

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            val_loss += loss.item()

        train_loss, val_loss = train_loss / len(train_dataloader), val_loss / len(val_dataloader)
        logger.info(f'Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.6f} Val loss: {val_loss:.6f}')
        wandb.log({"cl_training_loss": train_loss, "cl_validation_loss": val_loss}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, 'peft_encoder_weights/')
            model.model.save_pretrained(str(path))
            logger.info(f'Best model params saved at {save_dir}')


def setup_embedding_train(num_ssl_epochs,
                          save_dir=None,
                          num_triplets=None,  # Number of triplets will be equal to the size of the dataset
                          ssl_loss='triplet',
                          model_max_tokens=512,
                          target_r=14,
                          init_r=20,
                          lora_alpha=32,
                          lora_dropout=0.1,
                          model_name='intfloat/multilingual-e5-base',
                          device_type='cuda:0',
                          train_batch_size: int = 8,
                          eval_batch_size: int = 8,
                          learning_rate=3e-5,
                          weight_decay=0.01, ):
    assert ssl_loss in AVAILABLE_SSL_LOSSES, f'Not supported contrastive loss {ssl_loss}.' \
                                             f' Available: {AVAILABLE_SSL_LOSSES.keys()}'
    save_dir = _setup_save_dir(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(target_r, init_r, lora_alpha, lora_dropout, model_name).to(device)
    ssl_dataset = _compose_apn_dataset(model.tokenizer, model_max_tokens, num_triplets)
    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss]()

    train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay, device)
    logger.info('Training has finished')


def setup_classifier_train(num_cls_epochs,
                           num_ssl_epochs,
                           save_dir=None,
                           num_authors=100,
                           num_triplets=None,  # Number of triplets will be equal to the size of the dataset
                           cls_loss='arcface',
                           ssl_loss='triplet',
                           model_max_tokens=512,
                           target_r=14,
                           init_r=20,
                           lora_alpha=32,
                           lora_dropout=0.1,
                           model_name='intfloat/multilingual-e5-base',
                           device_type='cuda:0',
                           train_batch_size: int = 8,
                           eval_batch_size: int = 8,
                           learning_rate=3e-5,
                           gradient_accumulation_steps=4,
                           weight_decay=0.01,
                           ):
    assert ssl_loss in AVAILABLE_SSL_LOSSES, f'Not supported contrastive loss {ssl_loss}.' \
                                             f' Available: {AVAILABLE_SSL_LOSSES.keys()}'
    assert cls_loss in AVAILABLE_CLS_LOSSES, f'Not supported classification loss {cls_loss}. ' \
                                             f'Available: {AVAILABLE_CLS_LOSSES.keys()}'
    save_dir = _setup_save_dir(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(target_r, init_r, lora_alpha, lora_dropout, model_name).to(device)
    classification_dataset = _compose_dataset(model.tokenizer, model_max_tokens, num_authors)
    logger.info('Classification dataset created')
    cls_loss = AngularPenaltySMLoss(in_features=model.embedding_dim,
                                    out_features=num_authors, loss_type=cls_loss).to(device)

    train_classifier(model, classification_dataset, cls_loss, model.tokenizer, num_cls_epochs, save_dir,
                     train_batch_size, eval_batch_size, learning_rate, gradient_accumulation_steps, weight_decay)
    logger.info('Classification task training has finished, moving to APN training')

    tuned_ckpt_path = str(os.path.join(save_dir, 'peft_encoder_weights'))
    config = PeftConfig.from_pretrained(tuned_ckpt_path)
    model.model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model.model = PeftModel.from_pretrained(model.model, tuned_ckpt_path, is_trainable=True)
    logger.info('Loaded weights of best model from classification stage')

    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss]()
    logger.info('Creating the dataset for contrastive learning')
    ssl_dataset = _compose_apn_dataset(model.tokenizer, model_max_tokens, num_triplets)

    train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay, device)
    logger.info('Training has finished')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.models.embedders import EmbeddingModel
    from src.models.losses import AngularPenaltySMLoss

    wandb.init(project="yandex-camp-project")

    logger.info('Creating the experiment')
    Fire({
        'embedding': setup_embedding_train,
        'classifier': setup_classifier_train
    })
