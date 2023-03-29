# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

from utils import train, test, get_dataset
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
    parser.add_argument('--output_len', type=int, default=1, help='output len')
    parser.add_argument('--dropout', type=int, default=0.3, help='dropout')
    parser.add_argument('--window_size', type=int, default=168, help='window_size')
    parser.add_argument('--gcn_num', type=int, default=2, help='gcn_num')
    parser.add_argument('--glu_num', type=int, default=5, help='glu_num')
    parser.add_argument('--p', type=int, default=7, help='segment num')
    parser.add_argument('--runs', type=int, default=5, help='runs')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--use_gcn', type=bool, default=True, help='use gcn')
    parser.add_argument('--use_glu', type=bool, default=True, help='use glu')
    parser.add_argument('--use_te', type=bool, default=True, help='use te')
    parser.add_argument('--use_relu', type=bool, default=False, help='runs')

    ''' data '''
    parser.add_argument('--dataset_name', type=str, default='solar-energy', help='dataset name')
    parser.add_argument('--filter_num', type=int, default=24, help='runs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = args_parser()
    print(args)

    for m in [3, 6, 12, 24]:

        dataset = get_dataset(args, m)

        for n in range(args.runs):
            model = SGTANN(filter_num=args.filter_num, node_num=dataset[0][0].shape[2], window=args.window_size,
                           output_len=args.output_len, p=args.p, use_te=args.use_te, use_gcn=args.use_gcn,
                           dropout=args.dropout, use_relu=args.use_relu, use_glu=args.use_glu, gcn_num=args.gcn_num,
                           glu_num=args.glu_num)
            try:
                print('train')
                train(args, model, dataset, m)
            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

            result, y_test, y_test_pred, scale = test(args, model, dataset, m)

            gc.collect()
