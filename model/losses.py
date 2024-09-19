from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# for background reconstruction
from model.faceshifter.models.bisenet.bisenet import BiSeNet

# for perception loss
from torchvision.models import vgg19
from model.faceshifter.models.vgg.modules.vgg import VGG_Model

# for cx loss
from model.faceshifter.losses.loss import CXLoss

# for landmark loss
from model.H3R.torchalign import FacialLandmarkDetector_hzy

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
        
class face_mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.bisenet = BiSeNet(19)
        self.bisenet.load_state_dict(
            torch.load(
                "/data/huangziyao/projects/deepfake/cvpr2023/baseline/checkpoints_utils/bisenet/79999_iter.pth",
                map_location="cpu",
            )
        )
        self.bisenet.eval()

    def get_eye_mouth_mask(self, img):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest")
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 4).float())
        mask = mask + ((parsing == 5).float())
        mask = mask + ((parsing == 6).float())
        mask = mask + ((parsing == 11).float())
        mask = mask + ((parsing == 12).float())
        mask = mask + ((parsing == 13).float())

        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask
    
    def get_face_mask(self, img):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest")
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 14):
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask
    
    def get_head_mask(self, img):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest")
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 14):
            mask = mask + ((parsing == i).float())
        mask = mask + ((parsing == 17).float())
        mask = mask + ((parsing == 18).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask
    def forward(self, img, mask_type='face'):
        if mask_type=='face':
            return self.get_face_mask(img)
        elif mask_type=='eye_mouth':
            return self.get_eye_mouth_mask(img)
        elif mask_type=='head':
            return self.get_head_mask(img)

class background_rec_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
    def forward(self, tgt, swapped, face_mask):
        return self.loss( (1-face_mask)*tgt, (1-face_mask)*swapped )
        
class calac_attr_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, swap_attr, tgt_attr):
        att_loss = .0
        for i, (zat, yat) in enumerate(zip(swap_attr, tgt_attr)):
            att_loss += self.loss(zat, yat).mean()
        return att_loss

    
class calac_perecp_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.vgg_layer = [
            "conv_3_4",
            "conv_5_2",
            "conv_4_2",
            "conv_3_2",
            "conv_2_2",
            "conv_1_2",
            "conv_5_2",
        ]
        self.vgg_loss_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = vgg19(pretrained=False)
        vgg.load_state_dict(
            torch.load(
                "/data/huangziyao/projects/deepfake/cvpr2023/baseline/checkpoints_utils/vgg19-dcbb9e9d.pth",
                map_location="cpu",
            )
        )

        for param in vgg.parameters():
            param.requires_grad = False
        vgg.eval()
        
        self.vgg_model = VGG_Model(vgg, self.vgg_layer)
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        self.vgg_model.eval()
        self.l1 = nn.L1Loss()
        
    def forward(self, x, z):
        percep_loss = 0
        vgg19_features = self.vgg_model(torch.cat([x, z], dim=0))
        for ly, loss_weight in zip(
            ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
            self.vgg_loss_weights,
        ):
            x_feature, y_feature = vgg19_features[ly].chunk(2)
            percep_loss += self.l1(x_feature.detach(), y_feature) * loss_weight
        return percep_loss
    
class get_cx_loss(nn.Module):
    def __init__(self, loss_percep, reduction='mean'):
        super().__init__()
        
        self.loss_percep = loss_percep
        self.cxloss = CXLoss(0.2)
        # self.cxloss = CXLoss(0.1)
        
    def forward(self, tgt, swapped, eye_brow_mask_t):
        vgg_model = self.loss_percep.vgg_model
        
        eye_brow_out_mask_t = 1-eye_brow_mask_t
        
        cx_loss = 0
        vgg19_features = vgg_model(
            torch.cat(
                [
                    tgt * eye_brow_out_mask_t,
                    swapped * eye_brow_out_mask_t,
                ],
                dim=0,
            )
        )
        for ly in [
            "conv_3_4",
            "conv_4_2",
        ]:  # ["conv_3_2", "conv_3_4", "conv_4_2"]: 尝试一下每个layer的finetune效果
            # x target, y geenrated
            # x_resized = F.interpolate(vgg19_features[ly], [32, 32])
            x_resized = vgg19_features[ly]
            x_feature, y_feature = x_resized.chunk(2)
            cx_loss += self.cxloss(x_feature.detach(), y_feature).mean()
            
        return cx_loss
    
    
class get_landmark_loss(nn.Module):
    def __init__(self, model_path='/data/huangziyao/projects/deepfake/cvpr2023/baseline/model/H3R/models/lapa/hrnet18_256x256_p2/', reduction='mean'):
        super().__init__()
        self.model = FacialLandmarkDetector_hzy(model_path)
        self.model.eval()
        self.loss = nn.MSELoss(reduction=reduction)
            
    def forward(self, x, z):
        land1 = self.model(x)
        land2 = self.model(z)
        return self.loss(land1, land2)