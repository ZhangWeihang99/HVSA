# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

from .HVSA_Modules import *
import copy
from utils import cosine_similarity

class HVSA(nn.Module):

    def __init__(self, config={}, vocab_words=[]):
        super(HVSA, self).__init__()
        self.Eiters = 0
        self.extract_img_feature = EncoderImageFull(cfg=config)
        self.extract_text_feature = Skipthoughts_Embedding_Module(
            vocab=vocab_words,
            cfg=config
        )

        self.weight_loss = AutoWeightedModule(2)

    def forward(self, img, text, text_lens):
        if self.training is True:
            self.Eiters += 1

        batch_img = img.shape[0]
        batch_text = text.shape[0]

        # extract features
        img_feature = self.extract_img_feature(img)

        # text features
        text_feature = self.extract_text_feature(text, text_lens)
        dual_sim = cosine_similarity(img_feature.unsqueeze(dim=1).expand(-1, batch_text, -1),
                                     text_feature.unsqueeze(dim=0).expand(batch_img, -1, -1))

        return dual_sim, img_feature, text_feature


def factory(cfg, vocab_words, cuda=True):
    """
    get the model and set on device
    Args:
        vocab_words: all words sorted by frequency
    """

    cfg = copy.copy(cfg)

    # choose model
    if cfg['name'] == "HVSA":
        model = HVSA(cfg, vocab_words)
    else:
        raise NotImplementedError

    # use gpu
    if cuda:
        model.cuda()

    return model
