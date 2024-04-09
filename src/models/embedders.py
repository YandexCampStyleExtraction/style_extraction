import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

MODEL_TO_DIM = {
    'small': 384,
    'base': 768,
    'large': 1024
}


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, **input_batch):
        outputs = self.model(**input_batch)
        embeddings = average_pool(outputs.last_hidden_state, input_batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ClassifierBased(nn.Module):
    def __init__(self, model_name, num_classes, dropout_p=0.2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        input_dim = MODEL_TO_DIM[model_name.split('-')[-1]]
        classifier_neurons = [input_dim, 1024, 2048, 1024, num_classes]
        self.classifier = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(classifier_neurons[i - 1], classifier_neurons[i]),
                nn.BatchNorm1d(classifier_neurons[i]),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ) for i in range(1, len(classifier_neurons))])

    def forward(self, **input_batch):
        embeddings = self.model(input_ids=input_batch['input_ids'], attention_mask=input_batch['attention_mask'])
        embeddings = average_pool(embeddings.last_hidden_state, input_batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        pred = self.classifier(embeddings)
        return pred
