import sys
import torch.nn as nn

from .tmse import TMSE
from .ssws import TMSE_and_SSWS

class ActionSegmentationLoss(nn.Module):
    """
        Loss Function for Action Segmentation
        You can choose the below loss functions and combine them.
            - Cross Entropy Loss (CE)
            - Temporal MSE (TMSE)
    """

    def __init__(
        self, ce=True, tmse_ssws=True, tmse=False, weight=None, threshold=4, ignore_index=255,
        ce_weight=1.0, tmse_weight=0.15,ssws_tmse_weight=0.6
    ):
        super().__init__()
        self.criterions = []
        self.weights = []

        if ce: # 交叉熵损失函数                      https://zhuanlan.zhihu.com/p/77964128
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index))
            self.weights.append(ce_weight)

        #if tmse:            # 平滑损失函数
        #    self.criterions.append(
        #        TMSE(threshold=threshold, ignore_index=ignore_index))
        #    self.weights.append(tmse_weight)

        if tmse_ssws:            # 平滑损失函数
            self.criterions.append(
                TMSE_and_SSWS(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(ssws_tmse_weight)        

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds, gts, feats):   # 进行两次的损失函数计算 第一次是交叉熵损失函数 第二次为 平滑损失函数
        loss = 0.
        for criterion, weight in zip(self.criterions, self.weights):
            loss += weight * criterion(preds, gts)

        return loss
