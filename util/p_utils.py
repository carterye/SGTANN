import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MAE, MSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from project_utils import get_project_path
from tensorflow_addons.optimizers import AdamW
import time


def evaluate(true, pred, scale):
    def rae_np(actual, predicted):
        return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual - np.mean(actual)))

    def rrse_np(actual, predicted):
        return np.sqrt(np.mean(np.square(actual - predicted)) / np.mean(np.square(actual - np.mean(actual))))

    true_s = true * scale
    pred_s = pred * scale

    true_s = true_s.ravel()
    pred_s = pred_s.ravel()

    mae = round(mean_absolute_error(true_s, pred_s), 4)
    rmse = round(mean_squared_error(true_s, pred_s, squared=False), 4)

    rae = round(rae_np(true_s, pred_s), 4)
    rrse = round(rrse_np(true_s, pred_s), 6)

    sigma_t = true.std(axis=0)
    sigma_p = pred.std(axis=0)
    mean_t = true.mean(axis=0)
    mean_p = pred.mean(axis=0)
    index = (sigma_t != 0)
    corr_ = ((pred - mean_p) * (true - mean_t)).mean(axis=0) / (sigma_p * sigma_t)
    corr = round((corr_[index]).mean(), 4)
    result = {'rse': rrse, 'rae': rae, 'corr': corr, 'mae': mae, 'rmse': rmse}
    print(result)

    return result


def save_result(args, result, h):
    result['predict_mode'] = h
    result['model_name'] = args.model_name
    result['dataset_name'] = args.dataset_name
    result['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    path = get_project_path() + '/result/'
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + 'args.model_name' + '.csv'
    if not os.path.exists(path):
        res = pd.DataFrame(columns=result.keys())
    else:
        res = pd.read_csv(path)

    idx = len(res)
    for key in result.keys():
        res.loc[idx, key] = result[key]

    res.to_csv(path, index=False)


def load_data(args):
    """ 读取数据集 """
    path = get_project_path() + '/data/' + args.dataset_name + '/' + args.dataset_name + '.txt'
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    data_s = np.zeros(data.shape)
    scale = np.ones(data.shape[1])

    for i in range(data.shape[1]):
        scale[i] = np.max(data[:, i])
        data_s[:, i] = data[:, i] / scale[i]

    del data
    return data_s, scale


def sliding_window(args, data, ts, te, h):
    window_s = ts - h + 1 - args.window_size
    window_e = ts - h + 1
    target = ts

    # 滑动窗口生成样本
    x_data = []
    y_data = []
    while target < te:
        x_data.append(data[window_s: window_e])
        y_data.append(data[target])
        window_s += 1
        window_e += 1
        target += 1

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return [x_data, y_data]


def get_dataset(args, h):
    print(args.dataset_name, h)

    # 加载数据集
    data, scale = load_data(args)

    len_data = len(data)
    a = int(0.6 * len_data)
    b = int((0.6 + 0.2) * len_data)
    c = len_data

    # 滑动窗口
    train_set = sliding_window(args, data, args.window_size + h - 1, a, h)
    valid_set = sliding_window(args, data, a, b, h)
    test_set = sliding_window(args, data, b, c, h)

    print('Train Set: ', train_set[0].shape, train_set[1].shape)
    print('Val Set: ', valid_set[0].shape, valid_set[1].shape)
    print('Test Set: ', test_set[0].shape, test_set[1].shape)

    return train_set, valid_set, test_set, scale


def get_batches(inputs, targets, batch_size):
    length = len(inputs)
    index = tf.random.shuffle(tf.range(length))
    start_idx = 0
    while start_idx < length:
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        X = tf.gather(inputs, excerpt)
        Y = tf.gather(targets, excerpt)
        yield X, Y
        start_idx += batch_size


def train(args, model, dataset, h):
    train_set, valid_set, test_set, scale = dataset
    x_train, y_train = train_set
    x_val, y_val = valid_set
    x_test, y_test = test_set

    scale_tensor = tf.convert_to_tensor(scale, dtype=tf.float32)
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

    optimizer = AdamW(weight_decay=args.decay, learning_rate=args.learning_rate)
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    print(model.summary())

    path = get_project_path() + '/model_saved/' + args.model_name + '/' + args.dataset_name + '/' + str(
        h) + '/weights.ckpt'

    if args.loss == 'MAE':
        loss_func = MAE
    else:
        loss_func = MSE

    min_loss = 10000
    cur_test_loss = 0
    loss_list = []
    for epoch in range(1, args.epochs + 1):
        print('\nepoch:', str(epoch) + '/' + str(args.epochs))
        start_time = time.time()
        loss_total = 0
        count = 0
        for step, (x, y) in enumerate(get_batches(x_train_tensor, y_train_tensor, args.batch_size)):
            with tf.GradientTape() as tape:
                x_pred = model(x)
                scale_ = tf.broadcast_to(scale_tensor, [y.shape[0], y.shape[1]])
                loss = tf.reduce_mean(loss_func(y * scale_, x_pred * scale_))
                loss_total += loss
                count += 1
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if step % 100 == 0:
                print(step, round(float(loss), 4))

        print("\nval: ")
        y_val_pred = model.predict(x_val)
        val_result = evaluate(y_val, y_val_pred, scale)

        print("test:")
        y_test_pred = model.predict(x_test)
        test_result = evaluate(y_test, y_test_pred, scale)

        cur_loss = val_result['rse']
        cur_loss_ = test_result['rse']
        if cur_loss < min_loss:
            print(min_loss, '->', cur_loss, test_result['rse'])
            min_loss = cur_loss
            cur_test_loss = cur_loss_
            model.save_weights(path)

        end_time = time.time()
        elapsed_time = end_time - start_time

        lc = float(loss_total) / count
        loss_list.append(lc)
        print('total_loss:', round(lc, 4), round(min_loss, 4), round(cur_test_loss, 4))
        print("Elapsed time: %.2f seconds." % elapsed_time)

    return np.array(lc)

def test(args, model, dataset, h):
    train_set, valid_set, test_set, scale = dataset
    x_test, y_test = test_set
    path = get_project_path() + '/model_saved/' + args.model_name + '/' + args.dataset_name + '/' + str(
        h) + '/weights.ckpt'

    model.load_weights(path)
    y_test_pred = model.predict(x_test, batch_size=args.batch_size)
    test_result = evaluate(y_test, y_test_pred, scale)
    save_result(args, test_result, h)
    return test_result, y_test, y_test_pred, scale


if __name__ == '__main__':
    print('utils')
