# encoding:utf-8
"""
* HVSA
* By Zhang Weihang
"""

import os
import copy
import numpy as np
import controller
import utils
from dataset import data_handlers



def main(configs):

    # choose model
    vocab, vocab_word = data_handlers.make_vocab(configs)

    # Create dataset, model, criterion and optimizer
    test_loader = data_handlers.get_test_loader(vocab, configs)

    model = controller.get_model(configs, vocab_word)

    controller.load_model(configs, model)

    sims = controller.model_test(test_loader, model)
    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(sims)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(sims)
    rsum = r1t + r5t + r10t + r1i + r5i + r10i
    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n rsum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, rsum
    )

    print(all_score)

    utils.log_to_txt(
        contexts=all_score,
        filename="experiment/" + str(configs['GPU']) + "/" + "result/" + configs['experiment_name'] + ".txt"
    )

    return [r1i, r5i, r10i, r1t, r5t, r10t, rsum]

def update_configs_savepath(configs, k):
    updated_configs = copy.deepcopy(configs)

    updated_configs['optim']['resume'] = configs['logs']['ckpt_save_path'] + configs['experiment_name'] + "/" \
                                         + str(k) + "/" + configs['model']['name'] + '_best.pth.tar'

    return updated_configs

if __name__ == '__main__':
    configs = controller.get_config()

    # calc ave k results
    last_score = []
    if not os.path.exists("experiment/" + str(configs['GPU']) + "/" + "result/"):
        os.makedirs("experiment/" + str(configs['GPU']) + "/" + "result/")

    for k in range(configs['random_seed']['nums']):
        print("=========================================")
        print("Start evaluate random seed {}".format(configs['random_seed']['seed'][k]))

        # update the checkpoint path
        update_configs = update_configs_savepath(configs, k)

        # get one experiment result
        one_score = main(update_configs)
        last_score.append(one_score)
        
        print("Complete evaluate evaluate random seed {}".format(configs['random_seed']['seed'][k]))

    # average result
    print("\n================ Ave Score On {}-random verify) ============".format(configs['random_seed']['nums']))
    last_score = np.average(last_score, axis=0)
    names = ['r1i', 'r5i', 'r10i', 'r1t', 'r5t', 'r10t', 'rsum']
    for name, score in zip(names, last_score):
        print("{}:{}".format(name, score))
        utils.log_to_txt(
            contexts="{}:{}".format(name, score),
            filename="experiment/" + str(configs['GPU']) + "/" + "result/" + configs['experiment_name'] + ".txt"
        )

