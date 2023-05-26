"""
Implements the knowledge distillation loss
"""
from abc import get_cache_token
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from utils import batch_index_select
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math


class DistillDiffPruningFLOPsLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, keep_ratio=0.7, 
                 clf_weight=0, token_dis=False, print_mode=True):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.keep_ratio = keep_ratio
        self.count = 0
        self.token_dis = token_dis
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0

        self.flopsT = teacher_model.cal_flops()
        print(self.flopsT)

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, mask, out_pred_score, flops = outputs

        pred_loss = 0.0
        pred_loss = (flops.mean() - self.flopsT*self.keep_ratio).norm(1)

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t, token_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )
        
        token_kl_loss = 0.0
        if self.token_dis:
            B, N, C = token_pred.size()
            assert mask.numel() == B * N

            bool_mask = mask.reshape(B*N) > 0.5

            token_pred = token_pred.reshape(B*N, C)
            token_t = token_t.reshape(B*N, C)

            if mask.sum() < 0.1:
                token_kl_loss = token_pred.new(1,).fill_(0.0)
            else:
                token_t = token_t[bool_mask]
                token_pred = token_pred[bool_mask]
                token_kl_loss = F.kl_div(
                        F.log_softmax(token_pred, dim=-1),
                        F.log_softmax(token_t, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
        
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            if self.token_dis:
                self.token_distill_loss += token_kl_loss.item()
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss