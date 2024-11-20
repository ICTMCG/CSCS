from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# lr scheduler for training
def adjust_learning_rate_cos(optimizer, epoch, init_lr, max_epoch):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    # if args.cos:  # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    # else:  # stepwise lr schedule
    #     for milestone in args.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    # features: [bsz, n_views, f_dim]
    # `n_views` is the number of crops from each image
    # better be L2 normalized in f_dim dimension
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print(logits_mask)
        # print(logits_mask.shape)
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # need to be confirmed
        # mean_log_prob_pos = (mask * log_prob).sum(1) / max(mask.sum(1))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class face_cos_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        # self.loss = nn.CosineEmbeddingLoss(reduction=reduction) # 'none' | 'mean' | 'sum'
    # def forward(self, z1, z2):
        # y = torch.ones((z1.shape[0]), device=z1.device) # label 1 means z1 & z2 should be the same one. 这么想的话也可以考虑让不同人不像
        # print(y.shape)
        # return self.loss(z1, z2, y)
    
    def forward(self, z1, z2):
        return (1 - torch.sum(z1 * z2, dim=1) / (torch.norm(z1, dim=1) * torch.norm(z2, dim=1))).mean()

class pixel_level_change_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction) # 'none' | 'mean' | 'sum'
        # self.loss = nn.L1Loss()
    def forward(self, x1, x2):
        return self.loss(x1, x2) / 3
        # return self.loss(x1, x2)
