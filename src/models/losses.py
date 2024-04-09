from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, smooth=0.0):
        super().__init__()
        self.H = torch.nn.CrossEntropyLoss(label_smoothing=smooth, reduction='mean')

    def forward(self, predicted_sim, gt_sim, T: Union[float, nn.Parameter] = 0.08):
        return self.H(predicted_sim / T, gt_sim)


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
    eps = 1e-8
    """Uses all valid triplets to compute Triplet loss
    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=1.):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """computes loss value.
        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
        Returns:
          Scalar loss value.
        """
        # step 1 - get distance matrix (embeddings are normed at the end, so it is just dot product for cosine distance)
        # shape: (batch_size, batch_size)
        distance_matrix = embeddings @ embeddings.T

        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0

        # shape: (batch_size, batch_size, batch_size)
        mask = TripletMarginLoss.get_triplet_mask(labels)
        triplet_loss *= mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > TripletMarginLoss.eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + TripletMarginLoss.eps)

        return triplet_loss

    @staticmethod
    def get_triplet_mask(labels):
        """compute a mask for valid triplets
        Args:
          labels: Batch of integer labels. shape: (batch_size,)
        Returns:
          Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
          A triplet is valid if:
          `labels[i] == labels[j] and labels[i] != labels[k]`
          and `i`, `j`, `k` are different.
        """
        # step 1 - get a mask for distinct indices

        # shape: (batch_size, batch_size)
        indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
        indices_not_equal = torch.logical_not(indices_equal)
        # shape: (batch_size, batch_size, 1)
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        # shape: (1, batch_size, batch_size)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        # Shape: (batch_size, batch_size, batch_size)
        distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # step 2 - get a mask for valid anchor-positive-negative triplets

        # shape: (batch_size, batch_size)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # shape: (batch_size, batch_size, 1)
        i_equal_j = labels_equal.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        i_equal_k = labels_equal.unsqueeze(1)
        # shape: (batch_size, batch_size, batch_size)
        valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

        # step 3 - combine two masks
        mask = torch.logical_and(distinct_indices, valid_indices)

        return mask
