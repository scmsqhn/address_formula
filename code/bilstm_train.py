#!coding=utf8
import sys
import os
from tensorflow.python.ops import variable_scope as vs
import pdb
#import gensim
#from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import traceback
#import digital_info_extract as dex
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
import time
#import os
#import jieba
#import collections
#import sklearn.utils
#from sklearn.utils import shuffle
#import myconfig as config
import sklearn as sk
import tensorflow as tf
#rnn = tf.nn.rnn_cell
from tensorflow.contrib import rnn
#contrib import rnn
#import numpy as np
#import json
#import arctic
#from arctic import Arctic
#import pymongo
#import dmp.gongan.gz_case_address.predict as address_predict
#sys.path.append("/home/distdev/addr_classify")
CURPATH = os.path.dirname(os.path.realpath(__file__))
#import sys
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
print(CURPATH)
print(PARPATH)
#from bilstm import addr_classify
#from bilstm import eval_bilstm
sys.path.append(".")
sys.path.append("..")
#import bilstm
#from bilstm import datahelper
from datahelper import Data_Helper
#from eval_bilstm import Eval_Ner
#from bilstm import auto_encode
#from bilstm import text_cnn
#Eval
import logging
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
#from addr_classify.addr_classify import Addr_Classify
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEBUG =True
DATA = True
#import datetime
"""
envs = dict()
with open("envs.json", "r") as f:
    envs = json.loads(f.read())
"""
import sys
import const
Const = const._const()
Const.__setattr__("SUCC", "success")
Const.__setattr__("FAIL", "fail")
Const.__setattr__("FINISH", "finish")
Const.str2var()

def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(CURPATH, filepath)

def logging_init(loggername, filename):
    logger = logging.getLogger(loggername)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

bilstmlgr = logging_init("bilstm","./bilstm_logger.log")
datalgr = logging_init("data","./data_logger.log")
trainlgr = logging_init("train","./train_logger.log")
evallgr = logging_init("eval","./eval_logger.log")

lgrdct = {}
lgrdct['bilstmlgr']=bilstmlgr
lgrdct['datalgr']=datalgr
lgrdct['trainlgr']=trainlgr
lgrdct['evallgr']=evallgr

def _print(*l,name="bilstmlgr"):
    logger = lgrdct[name]
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

_print("\n cur dir file is ", CURPATH)
_print(Const.SUCC)
_print(Const.FAIL)

"""
#------- paraline struct init for 15001 15002 15003
# init the paral line struct
global NAME
global server
NAME = "server_15001"
#NAME = "server_15002"
#NAME = "server_15003"
cluster=tf.train.ClusterSpec({
    "worker": [
        "103.204.229.74:15001",#格式 IP地址：端口号，第1台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
        "103.204.229.74:15002",#格式 IP地址：端口号，第2台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:1
    ],
    "ps": [
        "103.204.229.74:15003",#格式 IP地址：端口号，第3台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:ps/task:0
    ]})
if NAME == "15001":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15002":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=1)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15003":
    global server
    server = tf.train.Server(cluster,job_name='ps',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
    server.join()

paral_saver = tf.train.Saver()
paral_summary_op = tf.merge_all_summaries()
paral_init_op = tf.initialize_all_variables()
paral_sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)
with paral_sv.managed_session(server.target) as sess:
    while 1:
        print sess.run([addwb,mutwb,divwb])

with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0',cluster=cluster)):
    # do sth here
"""
#------- paraline struct init end
from bilstm_att import Bilstm_Att
class Train_Bilstm_Ner(object):

    def __init__(self):
        _print("\ncls Train_Bilstm_Ner")
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement=True #FLAGS.allow_soft_placement
        tfconfig.log_device_placement=False #FLAGS.log_device_placement
        self.sess = tf.Session(config=tfconfig)
        self.datahelper = Data_Helper()
        self.model = Bilstm_Att(self.datahelper)
        self._lr = 1e-3
        _print(self.model.y_pred.shape)
        _print("\ncls Train_Bilstm_Ner init finish.")

    def compare_predict(self, tags, predict, text):
        return pd.DataFrame([tags, predict, text])

    def test_epoch(self):
        """Testing or valid."""
        _print("\n calcu the test_epoch")
        _batch_size = 10
        batch_num = 10
        fetches = [self.model.attention, self.model.accuracy, self.model.cost, self.model.train_op]
        #start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        for i in range(batch_num):
            _print("\n calcu the batch_num in test_epoch")
            X_batch, y_batch = self.datahelper.next_batch("eval")
            feed_dict = {self.model.X_inputs:X_batch, self.model.y_inputs:y_batch, self.model.lr:1e-5, self.model.batch_size:_batch_size, self.model.keep_prob:1.0}
            #pdb.set_trace()
            _att, _acc, _cost, _ = self.sess.run(fetches, feed_dict)
            _print("\n _att")
            _print(_att)
            _print("\n _acc, _cost")
            _print(_acc, _cost)
            _accs += _acc
            _costs += _cost
            _print("\n _accs, _costs")
            _print(_accs, _costs)
        _print("\n batch_num: ", batch_num)
        _print("\n acc 10个字一组，每组acc求和处以组数")
        mean_acc= _accs / batch_num
        mean_cost = _costs / batch_num
        return mean_acc, mean_cost

    """
    def fit(self, to_fetch)
        X_batch, y_batch = self.datahelper.next_batch()
        feed_dict = {self.model.X_inputs:X_batch, self.model.y_inputs:y_batch, self.model.lr:self.model._lr, self.model.batch_size:10, self.model.keep_prob:0.5}

    """
    def att_train(self):

        # Define Training procedure
        cnn = self.att_layer
        self.att_layer.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.att_layer.grads_and_vars = optimizer.compute_gradients(cnn.loss)
        self.att_layer.train_op = optimizer.apply_gradients(self.att_layer.grads_and_vars, global_step=self.att_layer.global_step)
        self.att_layer.optimizer = optimizer

    def train_step_att(self, x_bh, y_bh, sess):

            """
            A single training step
            """
            _print("\n> run train_step_att")

            fetch =  [self.att_layer.global_step, self.att_layer.train_op, self.att_layer.loss, self.att_layer.accuracy, self.att_layer.predictions]
            _feed_dict= {self.att_layer.input_x: x_bh, self.att_layer.input_y: y_bh, self.att_layer.dropout_keep_prob: 0.5}

            _print(_feed_dict)
            result = sess.run(fetch, feed_dict=_feed_dict)
            #_, _step, _loss, _accuracy = sess.run(fetch, feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #_print("{}: step {}, loss {:g}, acc {:g}".format(time_str, _l, _a))
            return  result

    def fit_train(self, sess):
        pass
        #self.att_layer = text_cnn.TextCNN( \
        #    sequence_length=32, \
        #    class_num=8, \
        #    vocab_size=len(data_helper.dct), \
        #    embedding_size=128,\
        #    num_filters=128,\
        #    filter_sizes=[2,3,4,5], \
        #    l2_reg_lambda=0.0)
        #self.att_layer.dropout_keep_prob = 0.5
        #self.model_combine()
        #ckpt = tf.train.get_checkpoint_state('./model/')
        #saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        saver = tf.train.Saver(max_to_keep=30)
        #model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        #saver.restore(sess,model_path)
        #print(self.model_path)
        #print(saver)
        sess.run(tf.global_variables_initializer())
        #test_fetches = [self.attention, self.accuracy, self.cost, self.train_op, self.y_pred]
        train_fetches = [self.attention, self.accuracy, self.cost, self.train_op, self.Precision,self.Recall,self.Fscore, self.TP, self.FP, self.TN, self.FN]
        #train_att_fetches = [self.att_layer.accuracy, self.att_layer.train_op, self.att_layer.predictions]
        #gen = self.datahelper.gen_train_data("train")
        for epoch in range(self.max_max_epoch):
            #self.datahelper.train_data_generator = self.datahelper.gen_train_data(per=0.8,name='train')
            #start_time = time.time()
            _costs, _accs, show_accs, show_costs  = 0.0, 0.0, 0.0, 0.0
            #_costs, _accs = 0.0, 0.0
            pass#pdb.set_trace()
            for batch in range(self.tf_batch_num):
                _print('EPOCH %d lr=%g' % (epoch+1, self._lr))
                _acc = 0.0
                X_batch, y_batch = self.batch_gen.__next__()
                pass#pdb.set_trace()
                feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:self.btsize, self.keep_prob:0.5}
                res_att, res_acc, res_cost, res_op, res_precision, res_recall, res_fscore, res_tp, res_fp,res_tn,res_fn \
                    = sess.run(train_fetches, feed_dict) # the self.cost is the mean self.cost of one batch
                #att_feed_dict={self.att_layer.input_x:self.textcnn_data_transform(_att,5), self.att_layer.input_y:tf.reshape(tf.one_hot(y_batch,1),(1000,8)), self.att_layer.dropout_keep_prob:0.5 }
                #_att_acc, _att_op, _att_pred, _ = sess.run(train_att_fetches, att_feed_dict) # the self.cost is the mean self.cost of one batch
                #_print(dict(zip(["_att_acc", "_att_op", "_att_pred"],[ _att_acc, _att_op, _att_pred])))
                self.insertParaDict("res_att", res_att)
                self.insertParaDict("res_acc", res_acc)
                self.insertParaDict("res_cost", res_cost)
                self.insertParaDict("res_op", res_op)
                self.insertParaDict("res_precision", res_precision)
                self.insertParaDict("res_recall", res_recall)
                self.insertParaDict("res_fscore", res_fscore)
                self.insertParaDict("res_tp", res_tp)
                self.insertParaDict("res_fp", res_fp)
                self.insertParaDict("res_tn", res_tn)
                self.insertParaDict("res_fn", res_fn)
                res_pred_tri_sum = np.sum(np.argmax(res_att.reshape(6400,3),1))
                res_pred_tri_mean = np.mean(np.argmax(res_att.reshape(6400,3),1))
                self.insertParaDict("res_pred_tri_sum",np.sum(res_pred_tri_sum))
                self.insertParaDict("res_pred_tri_mean",np.sum(res_pred_tri_mean))
                #_accs += _acc
                #_costs += _cost
                show_accs += res_acc
                show_costs += res_cost
                #_print("show_accs, show_costs, _accs, _costs")
                print("acc cost average")
                self.prtAllPara()
                if batch%100==1:
                   X_batch, y_batch = self.batch_gen.__next__()
                   feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:self.btsize, self.keep_prob:0.5}
                   res_mean_acc = show_accs/100
                   res_mean_cost = show_costs/100
                   self.insertParaDict("res_mean_acc", res_mean_acc)
                   self.insertParaDict("res_mean_cost", res_mean_cost)
                   #if mean_cost-self.basecost>0:
                   #    _print("\n> TRIGGER THE NEW _LR SETTING")
                   #    tvars=tf.trainable_variables()  # 获取模型的所有参数
                   #    grads,_=tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
                   #    optimizer=tf.train.AdamOptimizer(learning_rate=self._lr)   # 优化器
                   #    self.train_op=optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())
                   #self.mod_lr(mean_cost)
                   show_accs=0
                   show_costs=0
                if batch%100==1 and batch>100:
                   save_path = saver.save(sess, self.model_save_path, global_step=(epoch+1))
                   _print('the save path is ', save_path)
                   _print('***************save ok ***************')
                if batch%100==2 and batch>100:
                   self.datahelper.dct.save("./model/myDctBak")
                   _print("> my dct save ok")

    def fit_test(self):
        """
        this is for test
        """

    def fit_eval(self, sess):
        """
        this is for eval
        """
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        self.new_saver=tf.train.import_meta_graph(self.meta_graph_path)
        self.new_saver.restore(sess,self.model_path)
        #graph = tf.get_default_graph()
        self.X_inputs=tf.get_collection("model.X_inputs")[0]
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        self.y_pred_meta=tf.get_collection("model.y_pred")[0]
        self.lr=tf.get_collection("lr")[0]
        self.batch_size=tf.get_collection("batch_size")[0]
        self.keep_prob=tf.get_collection("keep_prob")[0]
        self.attention=tf.get_collection("attention")[0]
        self.correct_prediction_bilstm= tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.correct_prediction_attention = tf.equal(tf.cast(tf.argmax(self.y_pred_meta, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy_attention = tf.reduce_mean(tf.cast(self.correct_prediction_attention, tf.float32))
        self.accuracy_bilstm = tf.reduce_mean(tf.cast(self.correct_prediction_bilstm, tf.float32))
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(sess, tf.train.latest_checkpoint(self.model.checkpoint_path))
        X_batch, y_batch = self.batch_gen.__next__()
        test_fetches = [self.attention, self.accuracy_attention, self.accuracy_bilstm, self.y_pred_meta]
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:10, self.keep_prob:1.0}
        _att_pred, _att_acc, _bilstm_acc , _bilstm_pred = sess.run(test_fetches, feed_dict)
        print(_att_pred,_bilstm_pred, _att_acc, _bilstm_acc)
        return _att_pred,_bilstm_pred, _att_acc, _bilstm_acc


if __name__ == "__main__":
    _print("\n train.py")
    train_bilstm_ner_ins =  Train_Bilstm_Ner()
    #while(1):
    #    print(train_bilstm_ner_ins.model.batch_gen.__next__())
    #    pass#pdb.set_trace()
    #train_bilstm_ner_ins.att_train()
    #train_bilstm_ner_ins.test_train_step_att()
    #df = train_bilstm_ner_ins.get_arctic_df("dataframe", "gz_gongan_case_posseg_cut")
    #train_bilstm_ner_ins.data_helper.marker_the_addr_from_context()
    #    gen = train_bilstm_ner_ins.data_helper.gen_train_data()
    #    gen.__next__()
    train_bilstm_ner_ins.model.fit_train(train_bilstm_ner_ins.sess)
    #train_bilstm_ner_ins.model.fit_train(train_bilstm_ner_ins.sess, train_bilstm_ner_ins.datahelper)
    #train_bilstm_ner_ins.()

    #train_bilstm_ner_ins.model.init_eval_graph()
    #for i in range(2):
    #    _print("\n predict 1 sentence")
    #    train_bilstm_ner_ins.model.run()


