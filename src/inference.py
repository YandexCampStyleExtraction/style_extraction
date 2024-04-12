import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from fire import Fire
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding


def save_embeddings_to_file(peft_checkpoint_path, target_path, data_path=None, batch_size=8, device_type='cuda:0'):
    def get_birth(x):
        try:
            year = x.split()[-1].split('-')[0]
            return int(year)
        except:
            return -1

    def preprocess_function(examples):
        model_inputs = model.tokenizer(examples['text'], max_length=512, padding='max_length', truncation=True)
        model_inputs['labels'] = examples['labels']
        return model_inputs

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    device = torch.device(device_type)
    config = PeftConfig.from_pretrained(peft_checkpoint_path)

    model = EmbeddingModel(model_name=config.base_model_name_or_path)
    model.model = PeftModel.from_pretrained(model.model, peft_checkpoint_path, is_trainable=False)
    model.eval()
    model.to(device)

    if data_path is None:
        data_path = RAW_DATA_PATH

    authors = pd.read_csv(os.path.join(DATA_PATH, 'authors.csv'))
    authors['year_of_birth'] = authors.author.apply(get_birth)
    live_authors = authors[authors.year_of_birth > 0]
    live_authors_ids = live_authors.index

    data = pd.read_csv(data_path)
    data = data[data.author_id.isin(live_authors_ids)]

    author_year = pd.DataFrame({'author': live_authors.author.to_list(), 'year': live_authors.year_of_birth.to_list()})
    author_year.to_csv(str(os.path.join(target_path, 'author_year.csv')))

    dataset = Dataset.from_dict({"text": data.text, "labels": data.author_id})
    dataset = dataset.map(preprocess_function, batched=True, desc='Tokenizing data', remove_columns=['text'])
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)
    author2emb = {author_id: np.zeros(model.embedding_dim) for author_id in data.author_id.unique()}
    author2count = {author_id: 1 for author_id in data.author_id.unique()}
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=data_collator, drop_last=False, pin_memory=True, shuffle=False)
    for batch in tqdm(dataloader, desc='Computing embeddings'):
        batch = batch.to(device)
        labels = batch.pop('labels')
        with torch.inference_mode():
            pred = model(**batch).cpu().numpy()
        for i, label in enumerate(labels):
            author2count[label.item()] += 1
            author2emb[label.item()] += pred[i].reshape(-1)
    author2emb = {author_id: author2emb[author_id] / author2count[author_id] for author_id in author2emb.keys()}.items()
    author2emb = sorted(list(author2emb), key=lambda x: x[0])
    author2emb = np.array([x[1] for x in author2emb])
    np.savetxt(str(os.path.join(target_path, 'embeddings.txt')), author2emb)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.models.embedders import EmbeddingModel
    from src.train import DATA_PATH, RAW_DATA_PATH

    Fire(save_embeddings_to_file)
