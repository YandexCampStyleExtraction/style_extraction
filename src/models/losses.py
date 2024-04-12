from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, negative, temperature: Union[float, nn.Parameter] = 0.08):
        """
        Compute the contrastive loss.

        Args:
            anchor (torch.Tensor): The anchor embeddings, shape (batch_size, embedding_dim).
            positive (torch.Tensor): The positive embeddings, shape (batch_size, embedding_dim).
            negative (torch.Tensor): The negative embeddings, shape (batch_size, embedding_dim).
            temperature (float): The temperature parameter for the softmax. Default is 0.08.

        Returns:
            torch.Tensor: The contrastive loss, a single scalar value.
        """
        sim_matrix = torch.exp(torch.mm(anchor, positive.t()) / temperature)
        denominator = sim_matrix.sum(dim=1) + torch.exp(torch.mm(anchor, negative.t()) / temperature).sum(dim=1)
        loss = -torch.log(sim_matrix.diag() / denominator)
        return loss.mean()


# https://arxiv.org/pdf/1901.05903.pdf
# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        elif loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        elif loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        else:
            raise NotImplementedError(f'loss_type [{self.loss_type}] is not implemented!')
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W.data = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))
        else:
            raise NotImplementedError(f'loss_type [{self.loss_type}] is not implemented!')

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    @staticmethod
    def compute_distance_matrix(anchor, positive, negative):
        """
        Compute distance matrix between anchor, positive, and negative samples.
        """
        distance_matrix = torch.zeros(anchor.size(0), 3)
        distance_matrix[:, 0] = 1 - F.cosine_similarity(anchor, anchor)  # this is probably always zero?
        distance_matrix[:, 1] = 1 - F.cosine_similarity(anchor, positive)
        distance_matrix[:, 2] = 1 - F.cosine_similarity(anchor, negative)
        return distance_matrix

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss using the batch hard strategy.
        """
        distance_matrix = TripletMarginLoss.compute_distance_matrix(anchor, positive, negative)
        hard_negative = torch.argmax(distance_matrix[:, 2])
        loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + self.margin)
        loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + self.margin)
        return torch.mean(loss)
