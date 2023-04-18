# -*- coding: utf-8 -*-
import os
import sys
sys.path.extend(['/home/ycy/SGTANN/'])

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

from util.p_utils import train, test, get_dataset
from project_utils import get_project_path
from model.models import SGTANN
import argparse
import gc
import warnings

warnings.filterwarnings('ignore')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='my_model_f', help='model name')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.00001, help='decay')
    parser.add_argument('--loss', type=str, default='MAE', help='loss')
    parser.add_argument('--th', type=float, default=0, help='th')
    parser.add_argument('--output_len', type=int, default=1, help='output len')
    parser.add_argument('--dropout', type=int, default=0.3, help='dropout')
    parser.add_argument('--window_size', type=int, default=168, help='window_size')
    parser.add_argument('--gcn_num', type=int, default=2, help='gcn_num')
    parser.add_argument('--glu_num', type=int, default=5, help='glu_num')
    parser.add_argument('--p', type=int, default=7, help='segment num')
    parser.add_argument('--runs', type=int, default=8, help='runs')
    parser.add_argument('--epochs', type=int, default=40, help='epochs')
    parser.add_argument('--use_gcn', type=bool, default=True, help='use gcn')
    parser.add_argument('--use_glu', type=bool, default=True, help='use glu')
    parser.add_argument('--use_te', type=bool, default=True, help='use te')
    parser.add_argument('--use_relu', type=bool, default=False, help='runs')

    ''' data '''
    # parser.add_argument('--dataset_name', type=str, default='solar-energy', help='dataset name')
    parser.add_argument('--dataset_name', type=str, default='nature-gas1', help='dataset name')
    parser.add_argument('--filter_num', type=int, default=24, help='runs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    args = parser.parse_args()
    return args


def main(args):

    for h in [3, 6, 12, 24]:
    # for h in [3]:

        dataset = get_dataset(args, h)

        for n in range(args.runs):
            model = SGTANN(filter_num=args.filter_num, node_num=dataset[0][0].shape[2], window=args.window_size,
                           output_len=args.output_len, th=args.th, p=args.p, use_te=args.use_te, use_gcn=args.use_gcn,
                           dropout=args.dropout, use_relu=args.use_relu, use_glu=args.use_glu, gcn_num=args.gcn_num,
                           glu_num=args.glu_num)
            try:
                print('train')
                loss_list = train(args, model, dataset, h)
            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

            result, y_test, y_test_pred, scale = test(args, model, dataset, h)
            print(result)

            # save loss list
            # np.savetxt(get_project_path() + '/result/train/' + args.dataset_name + str(h) + '.txt', loss_list)

            gc.collect()


if __name__ == '__main__':
    args = args_parser()
    print(args)
    main(args)
    sys.exit(0)
