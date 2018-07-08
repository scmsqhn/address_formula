import sys
import os
import pdb
import gensim
from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import traceback
import numpy as np
import pandas as pd
import re
import time
import jieba
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
import arctic
import pymongo
CURPATH = os.path.dirname(os.path.realpath(__file__))
print("当前路径为", CURPATH)
sys.path.append(CURPATH)

"""
import bilstm
from bilstm import addr_classify
from bilstm import eval_bilstm
from bilstm import datahelper_bilstm
from datahelper_bilstm import Data_Helper
from eval_bilstm import Eval_Ner
from bilstm import text_cnn
"""
import logging
import datetime
import const

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEBUG =True
DATA = True

print(const.const.SUCC)
print(const.const.FAIL)

def _path(filepath):
    """
    将当前路径与文件名合成文件的绝对路径
    """
    return os.path.join(CURPATH, filepath)

def logging_init():
    """
    初始化logger
    """
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("/home/distdev/bilstm/logger.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

lgr = logging_init()
print("完成logger的初始化")

def _print(*l):
    """
    使用lgr打印log并输出
    """
    logger = lgr
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

class Bilstm_Att(object):
    """
    长短记忆 及 注意力层 模型
    """
    def __init__(self, data_helper, tags):
        self.init_model_para()
        self.init_model_struct()
        ckpt = tf.train.get_checkpoint_state('./model/')
        self.model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        """
        分类类别标签
        """
        self.tags = tags  #{'b':1,'i':2,'e':3,'s':4,'a':5,'d':6,'r':7,'v':0}# words bg mid end / addrs bg mid end
        self.rev_tags = dict(zip(self.tags.values(), self.tags.keys()))

    def tag_map(self, pred_lst_1d):#[0:7]
        """
        将预测结果转化为标签输出
        """
        #print(pred_lst_1d)
        _ = list(pred_lst_1d)
        return self.rev_tags[_.index(max(_))]

    def textcnn_data_transform(self, data, n):
        assert n%2 ==1
        m = n//2
        """
        input data is a (1000,8) array
        """
        assert data.shape == (1000,8)
        output = []
        for i in range(0,1000):
            for j in range(i-m,i+m):
                if j<0 or j>999:
                    output.extend([0.0]*8)
                else:
                    output.extend(data[j,:])
        #pdb.set_trace()
        #print(np.array(output).reshape(1000,8*(n-1)))
        return np.array(output).reshape(1000,8*(n-1))

    def init_model_para(self, vocab,max_len, input_size, embedding_size, hidden_size, layer_num, max_grad_norm,  lr):
        self.decay = 0.85
        self.timestep_size =  max_len
        self.vocab_size = vocab #100208# 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = input_size #64
        self.embedding_size = embedding_size #64# 字向量长度
        self.class_num = len(self.tags)
        self.hidden_size = hidden_size #128# 隐含层节点数
        self.layer_num = layer_num #2        # bi-lstm 层数
        self.max_grad_norm = max_grad_norm #5.0  # 最大梯度（超过此值的梯度将被裁剪）
        self.model_save_path = _path("model/bilstm.ckpt") # 模型保存位置
        self.checkpoint_path = _path("model")  # 模型保存位置
        _print(self.model_save_path)
        _print(self.checkpoint_path)
        self._lr = lr

    def init_embedding(self):
        with tf.variable_scope("embedding") as embedding_scope:
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
            embedding_scope.reuse_variables()

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def init_bi_lstm(self):
        """build the bi-LSTMs network. Return the self.y_pred"""
        # self.X_inputs.shape = [batchsize, self.timestep_size]  -  inputs.shape = [batchsize, self.timestep_size, embedding_size]
        inputs = tf.nn.embedding_lookup(self.embedding, self.X_inputs)
        # ** 1.构建前向后向多层 LSTM
        cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        # ** 2.初始状态
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope('bidirectional_rnn',reuse=tf.AUTO_REUSE) as bidirectional_rnn:
            bidirectional_rnn.reuse_variables()
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw', reuse=tf.AUTO_REUSE) as fw_scope:
                #fw_scope.reuse_variables()
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw', reuse=tf.AUTO_REUSE) as bw_scope:
                #bw_scope.reuse_variables()
                inputs = tf.reverse(inputs, [1])
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1,0,2])
            self.output = tf.reshape(output, [-1, self.hidden_size*2])
            _print("\n bi_lstm output shape ", self.output.shape)
            return self.output # [-1, self.hidden_size*2]

    def input_placeholder(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [10, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [10, self.timestep_size], name='y_input')

    def init_outputs(self):
        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(self.bilstm_output, softmax_w) + softmax_b
            #pdb.set_trace()
            _print(self.y_pred.shape)

    def init_full_conn_layer(self,inp,shape0,shape1,dtype=tf.float32):
        softmax_w = self.weight_variable([shape0, shape1])
        softmax_b = self.bias_variable([shape1])
        outp= tf.matmul(inp, softmax_w) + softmax_b
        return tf.cast(outp,dtype)

    def init_attention(self):
            pred = self.y_pred
            embed = self.embedding
            print(pred.shape)
            print(pred.embed)
            attention = tf.concat([pred,embed], 1)
            div = 5
            attention = attention.reshape(15+128,20)
            a = tf.sequence_mask([i for i in range(div+1,20-div+1)],20,dtype=tf.bool)
            b = tf.sequence_mask([i for i in range(0,20-div+1)],20,dtype=tf.bool)
            mask = b-a # 20个长度的句子,5个字一个滑动窗口,一次向右滑动一位
            attention = tf.boolean_mask(attention, mask)
            y_input_mask = tf.boolean_mask(self.y_input, mask)
            # ====== dnn for every 55555 words
            divide = 5*15
            att_mask = self.init_full_conn_layer(attention, divide, divide,tf.float32)
            # ====== 将维度缩小输出
            p = tf.reshape(att_mask,-1)
            y = tf.reshape(y_input_mask,-1)
            self.TP, self.FP, self.TN, self.FN ,self.Precision, self.Recall,self.Fscore = self.fscore(labels=y, logits=p)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention)) #self.attention))
            self.cost = tf.multiply(self.Fscore, self.loss)
            correct_prediction = tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return self.cost

    def init_placeholder(self):
        self._lr=1e-3
        self.X_inputs = tf.placeholder(tf.float32, [self.btsize, self.timestep_size*self.embedding_size], name='X_input')
        self.y_inputs = tf.placeholder(tf.int32, [self.btsize, self.timestep_size], name='y_input')
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

    def init_model_struct(self):
        self.input_placeholder()
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.init_embedding()
        self.bilstm_output = self.init_bi_lstm()
        self.init_outputs()
        self.init_attention()
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention))
        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)   # 优化器
        # 梯度下降计算
        self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.cast(tf.reshape(tf.argmax(self.y_pred,1), [-1]), dtype=tf.float32)))
        self.save_graph_meta()

    def save_graph_meta(self):
        tf.add_to_collection('model.y_pred', self.y_pred)
        tf.add_to_collection('model.y_pred', self.y_pred)
        tf.add_to_collection('model.X_inputs',self.X_inputs)
        tf.add_to_collection('model.y_inputs',self.y_inputs)
        tf.add_to_collection('batch_size',self.batch_size)
        tf.add_to_collection('lr', self.lr)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('attention', self.attention)

