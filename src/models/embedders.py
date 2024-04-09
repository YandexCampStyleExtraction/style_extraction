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
        self.embedding_dim = MODEL_TO_DIM[model_name.split('-')[-1]]

    def forward(self, **input_batch):
        embeddings = self.model(input_ids=input_batch['input_ids'], attention_mask=input_batch['attention_mask'])
        embeddings = average_pool(embeddings.last_hidden_state, input_batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
