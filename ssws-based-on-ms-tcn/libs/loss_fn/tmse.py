import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSE(nn.Module):   # 平滑损失函数
    """
        Temporal MSE Loss Function
        Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
        arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold=4, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')  #这个MSElosss损失函数 这个博客有讲 https://blog.csdn.net/hao5335156/article/details/81029791

    def forward(self, preds, gts):

        total_loss = 0.
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]  # bitch_size 不为1 时 去除补0

            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1),               # F.log_softmax 这个函数这个博客有讲 https://blog.csdn.net/hao5335156/article/details/80607732
                F.log_softmax(pred[:, :-1], dim=1)
            )

            loss = torch.clamp(loss, min=0, max=self.threshold**2)  # 取一定区间的数进行计算  https://blog.csdn.net/u013230189/article/details/82627375
            total_loss += torch.mean(loss)

        return total_loss / batch_size                              


class GaussianSimilarityTMSE(nn.Module):
    """
        Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(self, threshold=4, sigma=1.0, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')
        self.sigma = sigma

    def forward(self, preds, gts, feats):

        total_loss = 0.
        batch_size = preds.shape[0]
        for pred, gt, feat in zip(preds, gts, feats):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            feat = feat[:, torch.where(gt != self.ignore_index)[0]]

            # calculate gaussian similarity
            diff = feat[:, 1:] - feat[:, :-1]
            similarity = torch.exp(
                - torch.norm(diff, dim=0) / (2 * self.sigma**2)
            )

            # calculate temporal mse
            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1),
                F.log_softmax(pred[:, :-1], dim=1)
            )
            loss = torch.clamp(loss, min=0, max=self.threshold**2)

            # gaussian similarity weighting
            loss = similarity * loss

            total_loss += torch.mean(loss)

        return total_loss / batch_size
