import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.experiments
import common.utils
import common.paths
import common.summary
import importlib
import numpy as np
import copy
import time
from common.log import log
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Attack:
    """
    Attack a model.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser()
        # main arguments
        parser.add_argument('config_module', type=str)
        parser.add_argument('config_target_variable', type=str)
        parser.add_argument('config_attack_variable', type=str)
        parser.add_argument('-snapshot', action='store_true', default=False)
        parser.add_argument('-wait', action='store_true', default=False)

        return parser

    def main(self):
        """
        Main.
        """

        module = importlib.import_module(self.args.config_module)
        #print(dir(module))
        assert getattr(module, self.args.config_target_variable, None) is not None, self.args.config_target_variable
        assert getattr(module, self.args.config_attack_variable, None) is not None, self.args.config_attack_variable
        target_configs = getattr(module, self.args.config_target_variable)
        #print(target_configs)
        if not isinstance(target_configs, list):
            target_configs = [self.args.config_target_variable]
        for target_config in target_configs:
            assert getattr(module, target_config, None) is not None, target_config
        attack_configs = getattr(module, self.args.config_attack_variable)
        if not isinstance(attack_configs, list):
            attack_configs = [self.args.config_attack_variable]
        for attack_config in attack_configs:
            assert getattr(module, attack_config, None) is not None, attack_config


        def no_writer(log_dir, sub_dir=''):
            return common.summary.SummaryWriter(log_dir)

        for j in range(len(target_configs)):
            target_config = getattr(module, target_configs[j])
            print(target_configs[j])

            scores = np.zeros([3, 6, 1000])
            errors = np.zeros([3, 6, 1000])

            if True:
                for i in range(len(attack_configs)):
                    print(attack_configs[i])
                    attack_config = getattr(module, attack_configs[i])
                    program = common.experiments.AttackInterface(target_config, attack_config)
                    if i == 0:
                        p, l, e, pc, lc, ec, mind = program.read_results(test_clean=True)
                        if 1:
                            #assert e.shape[0] == 1000
                            errc = np.sum(ec.astype(np.int16))
                            print('Error clean (%): {}'.format(errc / 10.0))
                            thres = np.sort(lc * (1 - ec.astype(np.int16)))[1000 - int((1000 - errc) / 100)]
                            thres = -thres
                            dist = np.sqrt(-thres)
                            print('Threshold distance: %.2f'%(dist))
                            print('Minimum d_xy : %.2f'%(mind))
                    else:
                        p, l, e = program.read_results(test_clean=False)

                    i0 = int(i / 6)
                    i1 = i % 6
                    errors[i0, i1] = e.astype(np.int16)

                    scores[i0, i1] = -l


                err99 = errors * ((scores > thres).astype(np.int16))
                err99n = (1 - err99)
                err99m = 1 - (err99n[0] * err99n[1] * err99n[2])
                err99ms = np.sum(err99m, axis=1) / 10.0

            print('Robust Errors (%) for 99% TPR')
            avg = np.mean(err99ms)
            wcs = np.max(err99ms)
            print('L_inf (epsilon=0.3) : {}'.format(err99ms[0]))
            print('L_inf (epsilon=0.4) : {}'.format(err99ms[1]))
            print('L_2 (epsilon=3) : {}'.format(err99ms[2]))
            print('L_1 (epsilon=18) : {}'.format(err99ms[3]))
            print('L_0 (epsilon=15) : {}'.format(err99ms[4]))
            print('Adv Frames : {}'.format(err99ms[5]))
            print('Average : %.1f'%(avg))
            print('Worst Case : {}'.format(wcs))




if __name__ == '__main__':
    setup_seed(100)
    program = Attack()
    program.main()