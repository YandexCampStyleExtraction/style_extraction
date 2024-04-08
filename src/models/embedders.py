import torch.functional as F
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel(nn.Module):
    def __init__(self, model_name='intfloat/multilingual-e5-large', device='cuda:0', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, input_batch):
        # input_batch = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**input_batch)
        embeddings = average_pool(outputs.last_hidden_state, input_batch['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T) * 100
        return scores


class ClassifierBased(EmbeddingModel):
    pass
