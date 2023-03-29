from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from model.model_aux import Attention, GCNConv


class SGTANN(keras.Model):
    def __init__(self, filter_num=24, node_num=137, window=168, output_len=1, th=0, p=7, use_te=True,
                 use_gcn=True, dropout=0.3, use_relu=False, use_glu=True, gcn_num=2, glu_num=5):
        super(SGTANN, self).__init__(name='SGTANN')
        self.filter_num = filter_num
        self.node_num = node_num
        self.window = window
        self.output_len = output_len
        self.th = th
        self.P = p
        self.S = window // p
        self.layer_num = glu_num
        self.gcn_num = gcn_num
        self.dropout = dropout
        self.hidden_units = node_num
        self.use_relu = use_relu
        self.use_te = use_te
        self.use_gcn = use_gcn
        self.use_glu = use_glu

        if self.use_gcn:
            self.gcn_list = []
            for i in range(self.gcn_num):
                self.gcn_list.append(GCNConv(self.window, self.P))

        self.filter_convs = []
        self.gate_convs = []
        self.skip_convs = []
        self.residual_convs = []

        residual_channels = filter_num
        conv_channels = filter_num
        skip_channels = filter_num * 2
        end_channels = filter_num * 2
        kernel_size = 3
        dilation_list = [1, 3, 6, 12, 24]

        self.start_conv = layers.Conv2D(filters=residual_channels, kernel_size=(1, 1))
        remain_window = self.window
        self.skipS = layers.Conv2D(filters=skip_channels, kernel_size=(1, self.window))
        for j in range(self.layer_num):
            dilation_rate = dilation_list[j]
            remain_window = remain_window - dilation_rate * (kernel_size - 1)
            if self.use_glu:
                self.filter_convs.append(
                    layers.Conv2D(filters=conv_channels, kernel_size=(1, kernel_size), dilation_rate=dilation_rate))
                self.gate_convs.append(
                    layers.Conv2D(filters=conv_channels, kernel_size=(1, kernel_size), dilation_rate=dilation_rate))
            else:
                self.filter_convs.append(
                    layers.Conv2D(filters=conv_channels, kernel_size=(1, kernel_size), dilation_rate=dilation_rate))

            self.skip_convs.append(
                layers.Conv2D(filters=skip_channels, kernel_size=(1, remain_window)))
            self.residual_convs.append(
                layers.Conv2D(filters=residual_channels, kernel_size=(1, 1)))

        self.skipE = layers.Conv2D(filters=skip_channels, kernel_size=(1, remain_window))
        self.end_conv_0 = layers.Conv2D(filters=end_channels, kernel_size=(1, 1))
        self.end_conv_1 = layers.Conv2D(filters=1, kernel_size=(1, 1))

        if self.use_te:
            self.layer1 = Sequential([
                layers.LSTM(self.hidden_units, dropout=self.dropout, return_sequences=True),
                layers.LSTM(self.hidden_units, dropout=self.dropout, return_sequences=True),
            ])
            self.skip_atten1 = Attention()
            self.skip_atten2 = Attention()
            if use_relu:
                self.dense0 = Sequential([layers.Dense(node_num),
                                         layers.Activation('relu')])
            else:
                self.dense0 = layers.Dense(node_num)

    def get_An(self, inputs, th):
        tmp1 = tf.transpose(inputs, [0, 2, 1])
        tm = tf.matmul(tmp1, tf.transpose(tmp1, [0, 2, 1]))
        tmp2 = tf.norm(tmp1, axis=2, ord=2, keepdims=True)
        tmp3 = tf.matmul(tmp2, tf.transpose(tmp2, [0, 2, 1])) + 1e-7
        tmp4 = tf.divide(tm, tmp3)
        zero = tf.zeros_like(tmp4)
        An = tf.where(tmp4 < th, x=zero, y=tmp4)
        An = tf.nn.softmax(An, axis=1)
        return An

    def call(self, inputs, **kwargs):
        inputs_x = tf.transpose(inputs, perm=[0, 2, 1])

        Ans = []
        for x in tf.split(inputs, axis=1, num_or_size_splits=self.P):
            Ans.append(self.get_An(x, self.th))
        Ans = tf.stack(Ans, axis=1)

        x1 = inputs_x
        if self.use_gcn:
            for i in range(self.gcn_num):
                x1 = inputs_x + tf.nn.relu(self.gcn_list[i]([Ans, x1]))

        x_raw = tf.expand_dims(x1, axis=-1)
        x = self.start_conv(x_raw)
        skip = self.skipS(layers.Dropout(self.dropout)(x_raw))
        for i in range(self.layer_num):
            residual = x

            if self.use_glu:
                filter = self.filter_convs[i](x)
                gate = layers.Activation('sigmoid')(self.gate_convs[i](x))
                x = filter * gate
            else:
                x = layers.Activation('relu')(self.filter_convs[i](x))

            x = layers.Dropout(self.dropout)(x)
            s = self.skip_convs[i](x)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.shape[2]:, :]

        x = layers.Activation('relu')(self.skipE(x) + skip)
        x = layers.Activation('relu')(self.end_conv_0(x))
        x = self.end_conv_1(x)
        out = x[:, :, 0, 0]

        if self.use_te:
            x_2_ = self.layer1(inputs)
            x_2 = x_2_[:, -1, :]

            x_s = tf.reshape(x_2_, [-1, self.P, self.S, self.hidden_units])
            x_3 = tf.reshape(x_s, [-1, self.P, self.S * self.hidden_units])
            x_3 = self.skip_atten1(x_3)

            x_3 = tf.reshape(x_3, [-1, self.S, self.hidden_units])
            c1 = self.skip_atten2(x_3)

            x_2 = self.dense0(tf.concat([c1, x_2], axis=1))
            out = out + x_2

        return out
