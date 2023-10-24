# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import torch
import tensorboard_logger as tb_logger
import logging
import json

import utils
from dataset import data_handlers
import controller

def main(config):
    """
    train and val the model
    """

    vocab, vocab_word = data_handlers.make_vocab(config)

    # Create dataset, model, criterion and optimizer
    print("load the data {}".format(config['dataset']['image_path']))
    train_loader, val_loader = data_handlers.get_train_val_loaders(vocab, config)
    model = controller.get_model(config, vocab_word)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config['optim']['lr'])


    # optionally resume from a checkpoint
    if config['optim']['resume']:
        start_epoch = controller.load_model(config, model)
    else:
        start_epoch = 0

    # start to train
    best_rsum = 0
    best_score = ""
    for epoch in range(start_epoch, config['optim']['epochs']):
        # train for one epoch
        controller.model_train(train_loader, model, optimizer, epoch, config=config)
        # evaluate on validation set
        if epoch % config['logs']['eval_step'] == 0:
            rsum, all_scores = controller.model_validate(val_loader, model, config)
            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
                best_rsum = rsum
                # save ckpt
                utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'config': config,
                    },
                    is_best,
                    filename='best.pth.tar',
                    prefix=config['logs']['ckpt_save_path'],
                    model_name=config['model']['name']
                )
            print("Current random seed: {}".format(config['random_seed']['current_num']))
            print("Now  score:")
            print(all_scores)
            print("Best score:")
            print(best_score)


if __name__ == '__main__':

    config = controller.get_config()
    # make logger
    tb_logger.configure(config['logs']['logger_name'] +
                        config['experiment_name'] + '/', flush_secs=5)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    logging.info("HVSA CONFIG: \n {}".format(json.dumps(config, indent=4)))

    # k_fold verify
    for k in range(len(config['random_seed'])):
        random_seed = config['random_seed']['seed'][k]
        print("=========================================")
        print("Start with the random seed: {}".format(random_seed))
        utils.set_seed(random_seed)
        # generate random train and val samples
        data_handlers.generate_random_samples(config)
        # update save path
        update_config = utils.update_config(config, k)

        # run experiment
        main(update_config)
