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


class MLP(Chain):
    '''3層ニューラルネットワーク'''
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.layer1 = L.Linear(784, 256)
            self.layer2 = L.Linear(256, 64)
                
    def __call__(self, x):
        h1 = F.relu(self.layer1(x))
        h2 = self.layer2(h1)
        return h2
    
class Triplet_Network(Chain):
    '''Triplet Loss Wrapper'''
    def __init__(self, predictor):
        super(Triplet_Network, self).__init__(
            predictor = predictor
        )
        
    def __call__(self, anchor, positive, negative, semihard, margin=0.2):
        # 3種類のサンプルを順伝播
        anchor_embeddings = self.predictor(anchor)
        positive_embeddings = self.predictor(positive)
        negative_embeddings = self.predictor(negative)
        
        # semihard negative pairの選出
        if semihard:
            positive_dists = np.linalg.norm(anchor_embeddings.data - positive_embeddings.data, axis=1)
            negative_dists = np.linalg.norm(anchor_embeddings.data - negative_embeddings.data, axis=1)
            diff_dists = negative_dists - positive_dists
            # 0 < d(x_a - x_n) - d(x_a - x_p) < margin
            semihard_triplets_mask = np.logical_and(diff_dists>0, diff_dists<margin)
            if True in semihard_triplets_mask:
                anchor_embeddings = anchor_embeddings[semihard_triplets_mask]
                positive_embeddings = positive_embeddings[semihard_triplets_mask]
                negative_embeddings = negative_embeddings[semihard_triplets_mask]
            else:
                # semihardがないときは誤差が最小なペアを学習させる
                mask = np.argmin(diff_dists)
                anchor_embeddings = np.expand_dims(anchor_embeddings[mask], axis=0)
                positive_embeddings = np.expand_dims(positive_embeddings[mask], axis=0)
                negative_embeddings = np.expand_dims(negative_embeddings[mask], axis=0)
                
        # triplet lossを計算
        loss = F.triplet(anchor_embeddings, positive_embeddings, negative_embeddings, margin=margin)
        chainer.reporter.report({'loss': loss}, self)
        
        return loss
    
    def embedding(self, x):
        # embedding vectorを生成する
        return self.predictor(x)

    
class Triplet_Updater(training.StandardUpdater):
    '''
    iterator: 正常品サンプルのイテレータ
    optimizer: 最適化方法
    semihard: semihard negative pairのみで学習するフラグ
    negative_size: サンプリングする不良品のサンプル数
    margin: triplet lossのviolate margin
    '''
    def __init__(self, iterator, optimizer, sampling_method, semihard=False, negative_size=1, margin=0.2):
        super(Triplet_Updater, self).__init__(
            iterator,
            optimizer,
        )
        
        self.sampling_method = sampling_method
        self.semihard = semihard
        self.margin = margin
        self.negative_size = negative_size
        
    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = self.converter(batch, -1)
        anchor_batch = in_arrays
        batch_size = anchor_batch.shape[0]
        
        # 不良品をサンプリングする
        defective_sample = self.sampling_method(self.negative_size)
        
        # 正常品サンプルと不良品の全組み合わせ（デカルト積）マスク [正常品, 正常品, 不良品]
        cartesian = np.array(np.meshgrid(np.arange(batch_size), np.arange(batch_size), np.arange(defective_sample.shape[0]))).T.reshape(-1, 3)
        
        anchor_batch = anchor_batch[cartesian[:, 0]]
        positive_batch = anchor_batch[cartesian[:, 1]]
        negative_batch = defective_sample[cartesian[:, 2]]
        optimizer = self._optimizers['main']
        model = optimizer.target
        optimizer.update(model, anchor_batch, positive_batch, negative_batch, semihard=self.semihard, margin=self.margin)
