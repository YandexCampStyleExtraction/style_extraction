import datetime
import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, DatasetDict
from fire import Fire
from loguru import logger
from peft import get_peft_model, AdaLoraConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, DefaultDataCollator

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
        sampled_labels = np.random.choice(unique_labels, size=max_classes, replace=False)
        raw_df = raw_df[raw_df['author_id'].isin(sampled_labels)]

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

    train_text, train_labels = train_df['text'].to_list(), train_df['author_id'].to_list()
    val_text, val_labels = val_df['text'].to_list(), val_df['author_id'].to_list()
    test_text, test_labels = test_df['text'].to_list(), test_df['author_id'].to_list(),

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict({"text": train_text, "labels": train_labels})
    dataset['val'] = Dataset.from_dict({"text": val_text, "labels": val_labels})
    dataset['test'] = Dataset.from_dict({"text": test_text, "labels": test_labels})

    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc='Tokenizing data', remove_columns=['text'])
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
                                  collate_fn=data_collator, drop_last=True, pin_memory=True)
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
        for i, batch in tqdm(enumerate(val_dataloader), total=len(train_dataloader), leave=False, desc='Validating'):
            batch = batch.to(device)
            labels = batch.pop('labels')
            with torch.inference_mode():
                pred = model(**batch)
            loss = loss_fn(pred, labels)
            correctly_classified += (pred.argmax(dim=-1) == labels).sum().item()
            val_loss += loss.item()
        val_loss /= len(val_dataloader)
        accuracy = correctly_classified / (len(val_dataloader) * eval_batch_size)
        logger.info(f'Epoch={epoch + 1}/{epochs} | Train loss = {train_loss:.6f} |'
                    f' Val loss = {val_loss:.6f} | Val accuracy = {accuracy:.6f}')
        # Maybe do the logging over batches?
        wandb.log({"training_loss": train_loss, "validation_loss": val_loss,
                   "validation_accuracy": accuracy}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(os.path.join(save_dir, 'peft_encoder_weights'))
            logger.info(f'Best model params saved at {save_dir}')


def train_embeddings(model,
                     tokenized_dataset,
                     loss_fn,
                     epochs,
                     save_dir,
                     train_batch_size,
                     eval_batch_size,
                     learning_rate,
                     weight_decay):
    device = next(model.parameters()).device

    data_collator = DefaultDataCollator()

    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=train_batch_size,
                                  collate_fn=data_collator, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(tokenized_dataset['val'], batch_size=eval_batch_size,
                                 collate_fn=data_collator, drop_last=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader, desc='Training', leave=False):
            input_batch = {'input_ids': batch['input_ids'].to(device),
                           'attention_mask': batch['attention_mask'].to(device)}
            labels = batch['labels'].to(device)
            embeddings = model(**input_batch)

            loss = loss_fn(embeddings, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            input_batch = {'input_ids': batch['input_ids'].to(device),
                           'attention_mask': batch['attention_mask'].to(device)}
            labels = batch['labels'].to(device)
            with torch.no_grad():
                embeddings = model(**input_batch)

            loss = loss_fn(embeddings, labels)
            val_loss += loss.item()

        train_loss, val_loss = train_loss / len(train_dataloader), val_loss / len(val_dataloader)
        logger.info(f'Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.6f} Val loss: {val_loss:.6f}')
        wandb.log({"training_loss": train_loss, "validation_loss": val_loss}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(os.path.join(save_dir, 'peft_encoder_weights'))
            logger.info(f'Best model params saved at {save_dir}')


def setup_embedding_train(num_ssl_epochs,
                          num_authors=1000,
                          save_dir=None,
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
    _setup_save_dir(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(target_r, init_r, lora_alpha, lora_dropout, model_name).to(device)
    ssl_dataset = _compose_dataset(model.tokenizer, model_max_tokens, num_authors)
    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss]()

    train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay)
    logger.info('Training has finished')


def setup_classifier_train(num_authors,
                           num_cls_epochs,
                           num_ssl_epochs,
                           save_dir=None,
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
    _setup_save_dir(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(target_r, init_r, lora_alpha, lora_dropout, model_name).to(device)
    classification_dataset = _compose_dataset(model.tokenizer, model_max_tokens, num_authors)
    logger.info('Classification dataset created')
    cls_loss = AngularPenaltySMLoss(in_features=model.embedding_dim,
                                    out_features=num_authors, loss_type=cls_loss).to(device)

    train_classifier(model, classification_dataset, cls_loss, model.tokenizer, num_cls_epochs, save_dir,
                     train_batch_size, eval_batch_size, learning_rate, gradient_accumulation_steps, weight_decay)
    logger.info('Classification task training has finished, moving to APN training')
    for stage in STAGES:
        os.remove(os.path.join(DATA_PATH, f'{stage}.csv'))

    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss](distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    ssl_dataset = _compose_dataset(model.tokenizer, model_max_tokens, max_classes=None)

    train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay)
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
