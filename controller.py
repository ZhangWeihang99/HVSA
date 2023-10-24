# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import os
import argparse
import yaml
import time
import torch
import numpy as np
from torch.autograd import Variable
import tensorboard_logger as tb_logger
import logging

from loss import losses
import utils


def get_config():
    """
    get the parser to load the config
    """
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/HVSA_rsitmd.yaml', type=str,
                        help='path to a yaml options file')
    args = parser.parse_args()
    # load model config
    with open(args.config, 'r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    return config


def get_model(options, vocab_word):
    """
    get model HVSA

    """

    # choose model
    if options['model']['name'] == "HVSA":
        from layers import HVSA as models
    else:
        raise NotImplementedError

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])

    model = models.factory(options['model'], vocab_word, cuda=True,)

    return model


def model_train(train_loader, model, optimizer, epoch, config):
    """
    train one epoch

    """

    # extract value
    utils.adjust_learning_rate(config, optimizer, epoch)
    grad_clip = config['optim']['grad_clip']
    margin = config['optim']['alpha']
    # loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = config['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())

    # get the train data
    for i, train_data in enumerate(train_loader):
        images, captions, lengths, ids = train_data  #length: [22,19,19,18....] {list:128}

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)   #[128,3,256,256]
        input_text = Variable(captions)   #[128,22]

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_text = input_text.cuda()

        scores, image_feature, text_feature = model(input_visual, input_text, lengths)
        torch.cuda.synchronize()

        # adaptive alignment
        beta = config['loss']['adaptive_alignment']['beta']
        loss_triplet = losses.difficulty_weighted_loss(
            scores, input_visual.size(0), margin, beta) / (batch_size*batch_size)

        # feature uniformity
        gamma = config['loss']['feature_uniformity']['gamma']
        loss_unif_image = losses.feature_unif_loss(image_feature, gamma)

        # auto weighted module
        loss = model.weight_loss([loss_triplet.unsqueeze(0), loss_unif_image.unsqueeze(0)])

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        train_logger.update('L', loss.item())
        train_logger.update('dwh_loss', loss_triplet.item())
        train_logger.update('unif_loss', loss_unif_image.item())

        # set grad zero, compute the grad and update the params
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print training information
        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)


def model_validate(val_loader, model, config):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0]*len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        for (id, img, cap, key,l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images, lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text, model, shard_size=config['dataset']['batch_size_val']
                        , lengths=input_text_lengeth)
    end = time.time()


    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    rsum = (r1t + r5t + r10t + r1i + r5i + r10i)

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, rsum
    )

    # for tensorboard
    tb_logger.log_value('rsum', rsum, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)

    return rsum, all_score


def model_test(test_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(test_loader.dataset)

    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for (id, img, cap, key, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images, lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text, model, lengths=input_text_lengeth)

    end = time.time()
    print("calculate infer time:", (end-start)/len(test_loader.dataset))

    return d


def load_model(options, model):
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    return start_epoch


