import torch.nn as nn
import torch
import torch.nn.functional as F

'''
    Loss function
    1. Focal Loss
    2. Label Smoothing Loss
    3. F1 Loss
'''
# 1. Focal Loss
class FocalLoss(nn.Module) :
    def __init__(self, weight = None, gamma = 2., reduction = "mean") :
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true) :
        log_prob = F.log_softmax(y_pred, dim = -1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1-prob)**self.gamma) * log-prob, y_true, weight=self.weight, reduction=self.reduction)


# 2. Label Smoothing
class LabelSmoothingLoss(nn.Module) :
    def __init__(self, classes = 18, smoothing = 0.1, dim = -1) :
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, y_pred, y_true) :
        y_pred = y_pred.log_softmax(dim=self.dim)
        with torch.no_grad() :
            true_dist = torch.zeros_like(y_pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, y_true.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * y_pred, dim = self.dim))


# 3. F1 Loss  
class F1Loss(nn.Module) :
    def __init__(self, classes = 18, epsilon = 1e-7) :
        self.classes = classes
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true) :
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim = 1)

        tp = (y_true * y_pred).sum(dim = 0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim = 0).to(torch.float32)
        fp = ((1 - y_true) * y_pred)/sum(dim = 0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim = 0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) (precision + recall + self.epsilon)
        f1 = f1.clamp(min = self.epsilon, max = 1-self.epsilon)

        return 1 - f1.mean()

