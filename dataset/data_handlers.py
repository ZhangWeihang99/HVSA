# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import torch
from dataset.data import PairwiseDataset
from dataset.vocab import deserialize_vocab
import utils
import random


def collate_fn(data):

    # Sort the data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, tokens, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # PAD
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]
    return images, targets, lengths, ids


def get_one_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PairwiseDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_train_val_loaders(vocab, opt):
    train_loader = get_one_loader('train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_one_loader('val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_one_loader('test', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader


def make_vocab(options):
    # make vocab
    print("Making the vocab")
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]
    return vocab, vocab_word


def generate_random_samples(options):
    """
    splite training data for train and val

    """
    # load all anns
    caps = utils.load_from_txt(options['dataset']['data_path'] + 'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path'] + 'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5, (img_id + 1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos) * percent)]
    val_infos = all_infos[int(len(all_infos) * percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options['dataset']['data_path'] + 'train_caps_verify.txt', mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path'] + 'train_filename_verify.txt', mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, options['dataset']['data_path'] + 'val_caps_verify.txt', mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path'] + 'val_filename_verify.txt', mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

