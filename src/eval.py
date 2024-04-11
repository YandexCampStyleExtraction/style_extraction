from fire import Fire
import sys
import os
import random

from src import metrics
from loguru import logger
import numpy as np
import pandas as pd
import torch
from peft import get_peft_model, AdaLoraConfig, PeftModel, PeftConfig
from torch.utils.data import DataLoader
from datetime import datetime
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, DefaultDataCollator, AutoModel
from tqdm import tqdm

DATA_PATH = 'data/'
ROOT_SAVE_PATH = 'output/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'full_dataset.csv')
STAGES = ['train', 'val', 'test']


def get_metrics(model, dataloader, device):
    embeddings = []
    y_true = []
    for batch in tqdm(dataloader, desc='Eval'):
        input_batch = {'input_ids': batch['input_ids'].to(device),
                       'attention_mask': batch['attention_mask'].to(device)}
        labels = batch['labels'].to(device)

        embeddings.append(model(**input_batch).cpu().detach().numpy())
        y_true.extend(labels.cpu().detach().tolist())

    embeddings = np.vstack(embeddings)
    y_true = np.array(y_true)

    # tpr_at_fpr
    metrics.eval_metrics(y_true, embeddings, verbose=True)


def _prepare_train_test_split(max_classes):
    raw_df = pd.read_csv(RAW_DATA_PATH).dropna()
    unique_labels = raw_df['author_id'].unique()
    if max_classes is not None and len(unique_labels) > max_classes:
        # Randomly sample the unique labels
        sampled_labels = np.random.choice(unique_labels, size=max_classes, replace=False)
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

    # train_text, train_labels = train_df['text'].to_list(), train_df['author_id'].to_list()
    # val_text, val_labels = val_df['text'].to_list(), val_df['author_id'].to_list()
    test_text, test_labels = test_df['text'].to_list(), test_df['author_id'].to_list(),

    dataset = DatasetDict()
    # dataset['train'] = Dataset.from_dict({"text": train_text, "labels": train_labels})
    # dataset['val'] = Dataset.from_dict({"text": val_text, "labels": val_labels})
    dataset['test'] = Dataset.from_dict({"text": test_text, "labels": test_labels})

    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc='Tokenizing data', remove_columns=['text'])
    return tokenized_dataset


def main(checkpoint_path, model_name="intfloat/multilingual-e5-base", device_type="cuda:0"):
    model = EmbeddingModel(model_name)
    device = torch.device(device_type)

    model.to(device)
    tuned_ckpt_path = checkpoint_path
    config = PeftConfig.from_pretrained(tuned_ckpt_path)
    model.model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model.model = PeftModel.from_pretrained(model.model, tuned_ckpt_path, is_trainable=True)
    logger.info(f'Loaded weights of {checkpoint_path=}')

    eval_dataset = _compose_dataset(model.tokenizer, 512, max_classes=None)
    test_dataloader = DataLoader(eval_dataset['test'], batch_size=16,
                                  collate_fn=DefaultDataCollator(), drop_last=True, pin_memory=True)
    get_metrics(model, test_dataloader, device)

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.models.embedders import EmbeddingModel

    Fire(
        main
    )
