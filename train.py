import random
import numpy as np
import chainer
from chainer import cuda, Function, \
                    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from sklearn.metrics import recall_score, confusion_matrix
from model import *

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

reset_seed(0)
x_train = np.load('./data/train_image.npy')
y_train = np.load('./data/train_label.npy')
x_test = np.load('./data/test_image.npy')
y_test = np.load('./data/test_label.npy')
# 0平均にする
x_train_mean = x_train.mean(axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
# 正常品と不良品を分ける
normal_mask = np.logical_or.reduce([y_train==6, y_train==0, y_train==9])
defective_mask = np.logical_not(normal_mask)
x_train_normal = x_train[normal_mask] # 正常品
x_train_defective = x_train[defective_mask] # 不良品
# 正常不良ラベル
y_train_anomaly = np.logical_or.reduce([y_train==6, y_train==0, y_train==9]).astype(np.float32)
y_test_anomaly = np.logical_or.reduce([y_test==6, y_test==0, y_test==9]).astype(np.float32)


defective_idx = 0
np.random.shuffle(x_train_defective)

def get_defective_sample(sample_size):
    '''不良品をサンプリングする関数'''
    global defective_idx
    global x_train_defective

    defective_sample = x_train_defective[defective_idx:defective_idx+sample_size]
    defective_idx += sample_size
    if defective_idx >= x_train_defective.shape[0]:
        defective_idx = 0
        np.random.shuffle(x_train_defective)
    return defective_sample

best_specificity = 0
best_model = None

@training.make_extension(trigger=(1, 'epoch'))
def print_specificity(trainer):
    '''評価用自作chainer extention'''
    print('-'*50)
    
    global best_specificity
    global best_model
    
    # 正常品Embeddingの重心を基準ベクトルとする
    base_vector = model.embedding(x_train_normal).data.sum(axis=0) / x_train_normal.shape[0]

    # 評価セットをEmbedding
    x_test_embedding = model.embedding(x_test).data

    # 評価セットと基準ベクトルのユークリッド距離を計算
    distances = np.linalg.norm(x_test_embedding-base_vector, axis=1)

    # 閾値をサーチ
    for d in np.linspace(0.01, distances.max()+0.01, num=1000):
        y_pred_anomaly = (distances<d).astype(np.float32)
        recall = recall_score(y_test_anomaly, y_pred_anomaly)
        if recall >= 0.999:
            threshold = d
            print(f"threshold: {threshold}")
            print(f"recall: {recall}")

            # 真陰性率を計算
            tn, fp, fn, tp = confusion_matrix(y_test_anomaly, y_pred_anomaly).ravel()
            specificity = tn / (tn+fp)
            print(f"specificity: {specificity}")
            print('-'*50)
            if specificity > best_specificity:
                best_specificity = specificity
                best_model = model
            break
        
if __name__ == '__main__':
    print(f"batch_size: {32} | margin: {1}")
    # 訓練データイテレーター
    train_iter = iterators.SerialIterator(x_train_normal, 32, shuffle=True)

    # モデルのインスタンス化と学習
    model = Triplet_Network(MLP())
    model.cleargrads()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    updater = Triplet_Updater(train_iter, optimizer, sampling_method=get_defective_sample, semihard=True, negative_size=5, margin=1)
    trainer = training.Trainer(updater, (50, 'epoch'))
    trainer.extend(extensions.ExponentialShift('lr', 0.75), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(print_specificity)
    trainer.run()