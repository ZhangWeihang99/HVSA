# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import torch
from torch.autograd import Variable


def difficulty_weighted_loss(scores, size, margin, beta):
    """ Difficulty Weighted Loss inspired by CL"""

    diagonal = scores.diag().view(size, 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # triplet_loss indicates the difficulty of samples

    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # from easy to hard
    weight_s = torch.exp(-beta*cost_s)
    weight_im = torch.exp(-beta*cost_im)

    cost_s = cost_s * weight_s
    cost_im = cost_im * weight_im
    return cost_s.sum() + cost_im.sum()


def feature_unif_loss(x, gamma=2):

    """ get image features uniformity loss"""

    x = x[::5, :]
    x0 = x / x.norm(p=2, dim=1, keepdim=True)
    return torch.pdist(x0, p=2).pow(2).mul(-gamma).exp().mean()
