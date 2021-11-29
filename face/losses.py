import torch
from torch import nn


def get_loss(name, alpha):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    elif name == "lgmface":
        return LGMFace(alpha=alpha)
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine


class LGMFace(nn.Module):
    def __init__(self, alpha=0.01):
        super(LGMFace, self).__init__()
        self.alpha = alpha

    def forward(self, logit, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], logit.size()[1], device=logit.device)
        m_hot.scatter_(1, label[index, None], self.alpha)
        logit[index] *= (1 + m_hot)
        return logit, 1 / (1 + m_hot)

