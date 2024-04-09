import gc
import os
import random
import sys


import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from fire import Fire
from tqdm.auto import tqdm
from loguru import logger
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from fixtures import AVAILABLE_PEFT, AVAILABLE_CLS_LOSSES


def _get_peft_model(peft_type: str, model_name: str) -> nn.Module:
    assert peft_type in AVAILABLE_PEFT or peft_type == 'none', \
        f'Not supported PEFT method {peft_type}. Available: {AVAILABLE_PEFT.keys()}'
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


def _compose_self_supervised_dataset():
    return None


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        logger.info(f'Epoch={epoch+1}/{epochs} | Train loss = {train_loss:.6f} |'
                    f' Val loss = {val_loss:.6f} | Val accuracy = {accuracy:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.model.save_pretrained(os.path.join(save_dir, 'peft_encoder_weights'))
            torch.save(model.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
            logger.info(f'Best model (classifier and encoder) saved at {save_dir}')


def train_embeddings(model, dataset, loss, epochs, device):
    temperature = torch.tensor([0.08], requires_grad=True, device=device)
    # in train loop: torch.nn.utils.clip_grad_norm_(temperature, 1.0)


def setup_embedding_train(ssl_loss,
                          num_ssl_epochs,
                          peft_type='adalora',
                          model_name='intfloat/multilingual-e5-base',
                          device_type='cuda:0'):
    device = torch.device(device_type)
    model = _get_peft_model(peft_type, model_name, is_classifier=False).to(device)
    ssl_dataset = _compose_self_supervised_dataset()
    train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, device)


def setup_classifier_train(num_classes,
                           save_dir,
                           num_cls_epochs,
                           num_ssl_epochs,
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device(device_type)
    model = _get_peft_model(peft_type, model_name).to(device)
    classification_dataset = _compose_classification_dataset(model.tokenizer, model_max_tokens, num_classes)
    logger.info('Classification dataset created')
    cls_loss = AngularPenaltySMLoss(in_features=model.embedding_dim, out_features=num_classes, loss_type=cls_loss)

    train_classifier(model, classification_dataset, cls_loss, model.tokenizer, num_cls_epochs, save_dir,
                     train_batch_size, eval_batch_size, learning_rate, gradient_accumulation_steps, weight_decay)

    # ssl_dataset = _compose_self_supervised_dataset()
    # train_embeddings(model, ssl_dataset, ssl_loss, num_ssl_epochs, device)


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
