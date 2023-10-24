# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

# A revision version from Skip-thoughs
import skipthoughts

def factory(vocab_words, cfg , dropout=0.25):

    if cfg['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, cfg['type'])
        # vocab: ['list', 'of', 'words',]
        seq2vec = st_class(cfg['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=cfg['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec
