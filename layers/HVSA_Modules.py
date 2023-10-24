# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from layers import seq2vec

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l2norm(X, dim, eps=1e-8):
    """
    L2-normalize columns of X

    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class L2Norm(nn.Module):
    """
    input x [batch_size, embedding_size]
    return x [batch_size, embedding_size]
    """
    def forward(self, x):

        return x / x.norm(p=2, dim=1, keepdim=True)



class KEA(nn.Module):

    """ Key-Entity Attention"""

    def __init__(self, c, attention_channel=64):
        """

        :param c: int, the input and output channel number
        """

        super(KEA, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)
        self.attention_channel = attention_channel
        self.linear_0 = nn.Conv1d(c, self.attention_channel, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.attention_channel, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        idn = x
        # key
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)   # b*c*n

        # key * query
        attn = self.linear_0(x)  # b*k*n
        attn = F.softmax(attn, dim=-1)  # b*k*n
        attn = attn / (1e-9 + attn.sum(keepdim=True, dim=1))  # b*k*n

        # value
        x = self.linear_1(attn)
        x = x.view(b, c, h, w)
        x = self.conv2(x)

        # res-connetction
        x = x + idn
        x = F.relu(x)

        return x


class AutoWeightedModule(nn.Module):

    def __init__(self, num_loss):
        super(AutoWeightedModule, self).__init__()
        self.num_loss = num_loss
        self.linear = nn.Sequential(
            nn.Linear(num_loss, 64, bias=False),
            nn.PReLU(),
            nn.Linear(64, 128, bias=False),
            nn.PReLU(),
            nn.Linear(128, num_loss, bias=False),
        )

        self.softmax = nn.Softmax(dim=0)


    def forward(self, loss):
        # [num_loss, 1]

        loss = torch.cat(loss, dim=0)
        self.weight = self.softmax(self.linear(loss))
        return (self.weight*loss).sum()


class Skipthoughts_Embedding_Module(nn.Module):

    def __init__(self, vocab, cfg, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.cfg = cfg
        self.vocab_words = vocab

        self.seq2vec = seq2vec.factory(self.vocab_words, self.cfg['seq2vec'], self.cfg['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.cfg['embed']['embed_dim'])

        self.dropout = out_dropout

    def forward(self, input_text, text_len):
        x_t_vec = self.seq2vec(input_text, text_len)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out




class EncoderImageFull(nn.Module):

    def __init__(self, cfg={}):

        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = cfg['embed']['embed_dim']


        #Load a pre-trained model
        self.resnet = resnet18(pretrained=True)

        # Replace the last fully connected layer of CNN with a new one
        self.fc = nn.Linear(self.resnet.fc.in_features, cfg['embed']['embed_dim'])

        self.key_entity_atten = KEA(512)

    def forward(self, images):
        """Extract image feature vectors."""
        # features = self.cnn(images)
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)

        # use self attention
        f4 = self.resnet.layer4(f3)
        f4 = self.key_entity_atten(f4)


        features = self.resnet.avgpool(f4)
        features = torch.flatten(features, 1)
        # linear projection to the joint embedding space
        features = self.fc(features)

        return features
