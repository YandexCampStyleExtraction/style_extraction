import gc
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from fire import Fire
from loguru import logger
from peft import get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, DefaultDataCollator

from fixtures import AVAILABLE_PEFT, AVAILABLE_CLS_LOSSES, AVAILABLE_SSL_LOSSES


def _get_peft_model(peft_type: str, model_name: str) -> nn.Module:
    assert peft_type in AVAILABLE_PEFT or peft_type == 'none', \
        f'Not supported PEFT method {peft_type}. Available: {AVAILABLE_PEFT.keys()} and "none"'
    # The parameters setting would probably be much more good-looking if placed in .yaml or something
    base_model = EmbeddingModel(model_name)

    if peft_type == 'prompt':
        logger.warning('Prompt tuning provides very few trainable parameters.'
                       ' The performance is likely to be suboptimal.')
    elif peft_type == 'none':
        logger.warning('The whole model will be tuned. This may take a while.')
        return base_model

    peft_config = AVAILABLE_PEFT[peft_type]
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


def _compose_classification_dataset(tokenizer, model_max_tokens, num_classes):
    # This is just a sample for sanity checks, reformat
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['text'], max_length=model_max_tokens, padding='max_length', truncation=True)
        model_inputs['labels'] = examples['labels']
        return model_inputs

    input_texts = [np.random.choice(["This is an example text", "This is another example text"]) for _ in range(1024)]
    labels = np.random.randint(0, 10, size=1024)
    dataset_dict = {"text": input_texts, "labels": labels}

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict(dataset_dict)
    dataset['test'] = Dataset.from_dict(dataset_dict)

    tokenized_dataset = dataset.map(preprocess_function,
                                    batched=True,
                                    desc='Tokenizing dataset',
                                    remove_columns=['text'])
    return tokenized_dataset


def _compose_self_supervised_dataset(tokenizer, model_max_tokens, num_authors):
    # This is just a sample for sanity checks, reformat
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

    anchor = ["Here at index I should be a text by some author" for _ in range(1024)]
    positive = ["Here at index I should be a text by the same author as in anchor[i]" for _ in range(1024)]
    negative = ["Here at index I should be a text by author != author of anchor[i]" for _ in range(1024)]

    dataset_dict = {"anchor": anchor, "positive": positive, "negative": negative}

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict(dataset_dict)
    dataset['test'] = Dataset.from_dict(dataset_dict)

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
                                  collate_fn=data_collator, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=eval_batch_size,
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
        for i, batch in tqdm(enumerate(test_dataloader), total=len(train_dataloader), leave=False, desc='Validating'):
            batch = batch.to(device)
            labels = batch.pop('labels')
            with torch.inference_mode():
                pred = model(**batch)
            loss = loss_fn(pred, labels)
            correctly_classified += (pred.argmax(dim=-1) == labels).sum().item()
            val_loss += loss.item()
        val_loss /= len(test_dataloader)
        accuracy = correctly_classified / (len(test_dataloader) * eval_batch_size)
        logger.info(f'Epoch={epoch + 1}/{epochs} | Train loss = {train_loss:.6f} |'
                    f' Val loss = {val_loss:.6f} | Val accuracy = {accuracy:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(os.path.join(save_dir, 'peft_encoder_weights'))
            logger.info(f'Best model params saved at {save_dir}')


def train_embeddings(model,
                     tokenized_dataset,
                     loss_fn,
                     tokenizer,
                     epochs,
                     save_dir,
                     train_batch_size,
                     eval_batch_size,
                     learning_rate,
                     weight_decay):
    device = next(model.parameters()).device

    data_collator = DefaultDataCollator() # DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=train_batch_size,
                                  collate_fn=data_collator, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=eval_batch_size,
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
        for batch in tqdm(test_dataloader, desc='Validation', leave=False):
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

        train_loss, val_loss = train_loss / len(train_dataloader), val_loss / len(test_dataloader)
        logger.info(f'Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.6f} Val loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(os.path.join(save_dir, 'peft_encoder_weights'))
            logger.info(f'Best model params saved at {save_dir}')



def setup_embedding_train(save_dir,
                          num_ssl_epochs,
                          num_authors=1000,
                          ssl_loss='triplet',
                          model_max_tokens=512,
                          peft_type='adalora',
                          model_name='intfloat/multilingual-e5-base',
                          device_type='cuda:0',
                          train_batch_size: int = 8,
                          eval_batch_size: int = 8,
                          learning_rate=3e-5,
                          weight_decay=0.01, ):
    assert ssl_loss in AVAILABLE_SSL_LOSSES, f'Not supported contrastive loss {ssl_loss}.' \
                                             f' Available: {AVAILABLE_SSL_LOSSES.keys()}'
    device = torch.device(device_type)
    model = _get_peft_model(peft_type, model_name).to(device)
    ssl_dataset = _compose_self_supervised_dataset(model.tokenizer, model_max_tokens, num_authors)
    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss](distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    train_embeddings(model, ssl_dataset, ssl_loss, model.tokenizer, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay)
    logger.info('Training has finished')


def setup_classifier_train(num_classes,
                           save_dir,
                           num_cls_epochs,
                           num_ssl_epochs,
                           num_ssl_authors=1000,
                           cls_loss='arcface',
                           ssl_loss='triplet',
                           model_max_tokens=512,
                           peft_type='adalora',
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(peft_type, model_name).to(device)
    classification_dataset = _compose_classification_dataset(model.tokenizer, model_max_tokens, num_classes)
    logger.info('Classification dataset created')
    cls_loss = AngularPenaltySMLoss(in_features=model.embedding_dim,
                                    out_features=num_classes, loss_type=cls_loss).to(device)

    train_classifier(model, classification_dataset, cls_loss, model.tokenizer, num_cls_epochs, save_dir,
                     train_batch_size, eval_batch_size, learning_rate, gradient_accumulation_steps, weight_decay)
    logger.info('Classification task training has finished, moving to APN training')

    ssl_loss = AVAILABLE_SSL_LOSSES[ssl_loss](distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    ssl_dataset = _compose_self_supervised_dataset(model.tokenizer, model_max_tokens, num_ssl_authors)

    train_embeddings(model, ssl_dataset, ssl_loss, model.tokenizer, num_ssl_epochs, save_dir, train_batch_size,
                     eval_batch_size, learning_rate, weight_decay)
    logger.info('Training has finished')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.models.embedders import EmbeddingModel
    from src.models.losses import AngularPenaltySMLoss

    logger.info('Creating the experiment')
    Fire({
        'embedding': setup_embedding_train,
        'classifier': setup_classifier_train
    })
