# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import torch
import numpy as np
import sys
from torch.autograd import Variable
from collections import OrderedDict
import os
import random
import copy


def log_to_txt(contexts=None, filename="save.txt", mark=False, encoding='UTF-8', mode='a'):
    """save contexts as txt"""

    f = open(filename, mode, encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | " + str(contexts[c]) + "\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)

    f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)

    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def shard_dis(images, captions, model, shard_size=256, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            with torch.no_grad():
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                l = lengths[cap_start:cap_end]

                sim, _, _ = model(im, s, l)
                sim = sim.squeeze()
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        # get ground_truth's rank
        for i in range(5 * index, 5 * index + 5, 1):  # gt:i
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of t2i"""
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def save_checkpoint(state, is_best, filename, prefix='', model_name=None):
    """
    save the best checkpoint
    :param state: model information
    :param is_best: is or not best
    :param filename: the file name of the checkpoint
    :param prefix: the prefix of the save path
    :param model_name: current model 's name
    :return: None

    """
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix +model_name +'_best.pth.tar')
                # torch.save(state, prefix + filename)

        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(options, optimizer, epoch, warm_up=10):
    """
    Sets the learning rate to the initial LR
    decayed every n epochs

    """

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        
        if epoch / warm_up < 1:
            lr = options['optim']['lr'] * (epoch+1)/warm_up
        if epoch % options['optim']['lr_update_epoch'] == options['optim']['lr_update_epoch'] - 1:
            lr = lr * options['optim']['lr_decay_param']

        param_group['lr'] = lr

    print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))


def load_from_txt(filename, encoding="utf-8"):
    """
    read txt lines

    """
    f = open(filename, 'r', encoding=encoding)
    contexts = f.readlines()
    return contexts


def set_seed(seed=42):
    """
    set random seeds for reproduction of the code

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_config(config, k):
    """
    update the config

    """
    updated_config = copy.deepcopy(config)

    updated_config['random_seed']['current_num'] = k
    updated_config['logs']['ckpt_save_path'] = config['logs']['ckpt_save_path'] + \
                                                config['experiment_name'] + "/" + str(k) + "/"
    return updated_config
