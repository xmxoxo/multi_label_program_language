#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
多标签模型例子

数据集：program_language_Multil_label_data


'''

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from keras.models import Model, load_model, model_from_json
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Dense
from keras.utils.np_utils import to_categorical


# 数据预处理
def datapreprocess():
    # 训练数据预处理
    print('训练数据预处理...')
    outfile = './data/train.tsv'
    if os.path.exists(outfile):
        print('训练数据已存在,跳过处理过程。')
        return 0


    data_path = r'F:\dataset\multi_label\program_language_Multil_label_data_\Multil_label_data'
    train_file = os.path.join(data_path, 'train.tsv')
    val_file = os.path.join(data_path, 'validation.tsv')
    test_file = os.path.join(data_path, 'test.tsv')
    # 数据字段： title	tags
    df_train = pd.read_csv(train_file, sep='\t')
    df_val = pd.read_csv(val_file, sep='\t')
    df_test = pd.read_csv(test_file, sep='\t')
    
    # 提取所有标签
    all_tags = df_train['tags'].values.tolist() + df_val['tags'].values.tolist()
    all_tags = [json.loads(l.replace('\'', '\"')) for l in all_tags]
    print('all tags: %d\n %s' % (len(all_tags), all_tags[:10]))
    all_width = [len(x) for x in all_tags]
    print('max width:%d' % max(all_width))

    # 按出现的顺序汇总 tag 
    from functools import reduce
    f = lambda x,y:x+[y] if not y in x else x
    labels = reduce(lambda x,y:reduce(f,y,x), all_tags, [])

    '''
    # 传统的汇总代码
    labels = []
    for tags in all_tags: 
        for tag in tags:
            if not tag in labels:
                labels.append(tag)
    '''
    total_labels = len(labels)
    print('all labels: %d, %s' % (total_labels, labels[:10]))

    # 保存tag字典
    np_labels = np.array(labels)
    np.save('./data/tags.npy', np_labels)
    '''
    dict_oldcol = {'index':range(len(labels)) ,'values':labels}
    df_dict = pd.DataFrame(dict_oldcol)
    df_dict.to_csv(os.path.join('./data/', 'all_tags.csv'), index=0)
    '''

    print('正在生成label...')
    # tags字段做one-hoe编码 得到 label
    def onehot(tags, total_labels=total_labels):
        tag = json.loads(tags.replace('\'', '\"'))
        value = [labels.index(x) for x in tag]
        v = to_categorical(value, num_classes=total_labels, dtype=int)
        label = v.sum(axis = 0)
        return label.tolist()
    
    df_train['label'] = df_train['tags'].apply(onehot)
    df_val['label'] = df_val['tags'].apply(onehot)
    df_test['label'] = df_test['title'].apply(lambda x:[0]*num_labels)
    
    df_train.drop(['tags'], axis=1, inplace=True)
    df_val.drop(['tags'], axis=1, inplace=True)
    print(df_train.head())

    # 保存处理好的数据文件
    df_train.to_csv('./data/train.tsv', index=0, sep='\t')
    df_val.to_csv('./data/val.tsv', index=0, sep='\t')
    df_test.to_csv('./data/test.tsv', index=0, sep='\t')
    
    print('数据预处理完成。')


# 训练参数
learning_rate = 2e-5
epochs = 3
max_length = 128
batch_size = 32
# 标签数量
num_labels = 100

# BERT 预训练模型 
if os.name=='nt':
    bert_path = r'F:\models\uncased_L-12_H-768_A-12'
else:
    bert_path = '/mnt/sda1/models/uncased_L-12_H-768_A-12/'

config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path =  os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')
#-----------------------------------------
# bert4keras
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        for is_end, (title, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(title, maxlen=max_length)

            labels = json.loads(label)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 加载数据
def load_data(filename):
    df = pd.read_csv(filename, sep='\t')
    if gblSim:
        #df = df[-32:]
        df = df[-1024:]
    return df.values.tolist()

# 模型评价
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss

def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro= f1_score(y_val, predicted, average='macro', zero_division=0 )
    f1_score_micro = f1_score(y_val, predicted, average='micro', zero_division=0)
    f1_score_weighted = f1_score(y_val, predicted, average='weighted', zero_division=0)
    h_loss = 1-hamming_loss(y_val, predicted)

    print("accuracy:",accuracy)
    print("f1_score_macro:",f1_score_macro)
    print("f1_score_micro:",f1_score_micro)
    print("f1_score_weighted:",f1_score_weighted)
    print("hamming_loss:",h_loss)
    print()

# 计算每一个标签AUC
def measure_auc(label, pred, np_labels=None):
    auc = [roc_auc_score(label[:,i], pred[:,i]) for i in list(range(num_labels))]
    if not np_labels:
        np_labels = list(range(num_labels))
    #dict_dat = dict(zip(np_labels, auc))
    df = pd.DataFrame({"label_name":np_labels, "auc":auc})
    return df

# 组合模型类
class LabelModel():
    def __init__(self, models_path=''):
        self.model = None
        # 模型目录
        self.models_path = models_path

        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None

    # 组合模型
    def build_model(self, preload=0):
        # 加载BERT预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            return_keras_model=False,
        )
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(units=num_labels, activation='sigmoid')(output)
        model = Model(bert.model.input, output)
        self.model = model

        if preload:
            self.load_weight()
    
    def load_weight(self):
        weight = os.path.join(self.models_path, 'best.h5')
        if os.path.exists(weight):
            self.model.load_weights(weight)
            print('模型权重已加载.')

    def load_train_dat(self):
        train_file = './data/train.tsv'
        val_file = './data/val.tsv'

        # 加载数据集 tnews_train.tsv ocnli_train
        train_data= load_data(train_file)
        val_data= load_data(val_file)

        self.train_generator = data_generator(train_data, batch_size)
        self.valid_generator = data_generator(val_data, batch_size)

        if gblSim:
            print(type(train_data))
            print(len(train_data))
            print(train_data[:5])
            '''
            i = 0
            for dat in self.train_generator:
                print(dat) #[:5]
                i+=1
                if i>=5:break;
            '''
    # 加载待预测数据
    def load_predict_dat(self):
        datfile = './data/test.tsv'
        self.test_data = load_data(datfile)
        self.test_generator = data_generator(self.test_data, batch_size)

        if gblSim:
            print(type(self.test_data))
            print(len(self.test_data))
            print('self.test_data:', self.test_data[:5])
        '''
            print('predict sample:')
            i = 0
            for dat in self.test_generator:
                print(list(dat)) #.tolist())
                i+=1
                if i>=15:break;

            print('-'*40)
        '''

    # 训练模型
    def train(self):
        # categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
        self.model.compile(loss='binary_crossentropy', 
            optimizer=Adam(learning_rate=learning_rate), 
            metrics=['accuracy']) #categorical_accuracy  binary_accuracy

        #self.model.summary()
        checkpoint = ModelCheckpoint(os.path.join(self.models_path, 'best.h5'),
            monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)

        # 加载数据集进行训练
        self.load_train_dat()
        print('正在训练模型...')
        history_fit = self.model.fit_generator(
                self.train_generator.forfit(),
                steps_per_epoch=len(self.train_generator),
                epochs = epochs,
                validation_data=self.valid_generator.forfit(),
                validation_steps=len(self.valid_generator),
                callbacks = [checkpoint] #,  evaluator
                )

        print('正在保存训练曲线数据...')
        with open(os.path.join(self.models_path, 'model_fit.log'), 'w') as f:
            f.write(str(history_fit.history))
        print('训练曲线数据保存完成...')

    # 预测数据
    def predict(self):
        self.load_predict_dat()
        # 预测
        lst_predict = self.model.predict_generator(
            self.test_generator.forfit(random=False), 
            steps=len(self.test_generator), 
            verbose=1)

        # 预测结果是概率，要转成0和1
        lst_predict = lst_predict.round().astype(int)
        # 加载字典
        np_labels = np.load('./data/tags.npy').tolist()
        #print('np_labels:',np_labels)
        dat_pridict = [[np_labels[y] for y in np.where(np.array(x)==1)[0].tolist()] for x in lst_predict]

        # 保存预测结果
        datfile = os.path.join(self.models_path, 'predict.tsv')
        title = np.array(self.test_data)[:,0]
        dict_col = {'title':title ,'tags':dat_pridict}
        df_out = pd.DataFrame(dict_col)
        df_out.to_csv(datfile, index=0)
    
        #with open(datfile, 'w', encoding='utf-8') as f:
        #    f.write('\n'.join(map(str,dat_pridict)))
        print('预测结果已保存。')

    # 在验证集上评估模型
    def doeval(self):
        print('正在评估模型...')
        self.load_train_dat()

        # 取y_true
        i = 0
        y_val = []
        for x in self.valid_generator.forfit(random=False):
            y_val.extend(x[1].tolist())
            i+=1
            if i==len(self.valid_generator): break
        #print('y_val:%d\n %s', (len(y_val),y_val))
        
        # 预测
        lst_predict = self.model.predict_generator(
            self.valid_generator.forfit(random=False), 
            steps=len(self.valid_generator), 
            verbose=1)

        predicted = lst_predict.round().astype(int)
        #print('predicted:', predicted)
        print_evaluation_scores(y_val, predicted)

        # 标签AUC
        np_labels = np.load('./data/tags.npy').tolist()  
        df_auc = measure_auc(np.array(y_val), np.array(predicted), np_labels)
        datfile = os.path.join(self.models_path, 'measure_auc.tsv')
        df_auc.to_csv(datfile, index=0, sep='\t')
        print("val set mean column auc:", df_auc.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多标签模型训练工具')
    parser.add_argument('--task', default='train', type=str, help='train or predict')
    parser.add_argument('--models', default='./models/', type=str, help='model path')
    parser.add_argument('--epochs', default=20, type=int, help='epochs=20')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size=32')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning_rate=2e-5')
    parser.add_argument('--sim', default=0, type=int, help='simulation=0')
    args = parser.parse_args()

    task = args.task
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    models_path = args.models

    # 模拟状态
    gblSim = args.sim
    #if gblSim: epochs = 3

    print('运行参数: 模拟=%s, task=%s, models=%s, epochs=%d, batch_size=%d, learning_rate=%f' % (
        gblSim, task, models_path, epochs, batch_size, learning_rate))

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    datapreprocess()

    if task=='train':
        # 训练模型
        combo = LabelModel(models_path=models_path)
        combo.build_model()
        combo.train()
        combo.load_weight()
        combo.predict()
        combo.doeval()

    if task=='predict':
        combo = LabelModel( models_path=models_path)
        combo.build_model(preload=1)
        combo.predict()

    if task=='eval' :
        combo = LabelModel( models_path=models_path)
        combo.build_model(preload=1)
        combo.doeval()
    
