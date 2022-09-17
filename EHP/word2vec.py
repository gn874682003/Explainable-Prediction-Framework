""" A PyTorch implementation of CBOW word embedding mechanism. """

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from sklearn.manifold import TSNE

import Frame.Model as FM
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import gensim.models.word2vec as wv
from matplotlib.font_manager import *
from gensim.models.callbacks import CallbackAny2Vec

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(CBOW, self).__init__()
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)#, padding_idx=0
        self.linear_1 = nn.Linear(embed_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_data):
        embeds = self.embed_layer(input_data.view(input_data.size(0), input_data.size(1)))#).view(-1)
        #词向量相加
        embedsum = embeds[:,0,:]
        for i in range(1, embeds.size(1)):
            embedsum += embeds[:,i,:]
        output = F.relu(self.linear_1(embedsum))
        output = F.log_softmax(self.linear_2(output))
        return output

# Helper function
def context_to_tensor(context, idx_dict):
    """ Converts context list to tensor. """
    context_idx = [idx_dict[word] for word in context]
    return Variable(torch.LongTensor(context_idx))

def TestMetric(X_Test, Y_Test, model, type):
    pred_y = []
    true_y = []
    for x, y in zip(X_Test, Y_Test):
        x = torch.tensor(x, dtype=torch.int)
        prediction = model(x)  # rnn output
        _, pred = torch.max(prediction, 1)
        for ty, py in zip(y.numpy().tolist(), np.round(pred).numpy().tolist()):
            true_y.append(ty)
            pred_y.append(py)
    if type == 1:
        Metric = accuracy_score(true_y, pred_y)
    else:
        Metric = mean_absolute_error(true_y, pred_y)
    return Metric

def word2vec(Train_X, Train_Y, ConvertReflact):
    EMBED_DIM = 5
    olen = len(ConvertReflact[0])
    while olen > 20:
        olen /= 4 #5
        EMBED_DIM += 5
    HIDDEN_SIZE = 64
    LR = 0.001
    NUM_EPOCHS = 20

    vocab = ConvertReflact[0]
    model = CBOW(len(vocab), EMBED_DIM, HIDDEN_SIZE)
    # print(model.embed_layer.weight)
    loss_function = nn.NLLLoss()
    optimizer = opt.Adam(model.parameters(), lr=LR)

    # Training loop
    for e in range(NUM_EPOCHS):
        total_loss = torch.FloatTensor([0])
        for x,y in zip(Train_X,Train_Y):
            x = torch.tensor(x, dtype=torch.int)
            y = torch.tensor(y, dtype=torch.long)
            model.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        # Bookkeeping
        # if e % 10 == 0:
        # print('Epoch: %d | Loss: %f.4' % (e, total_loss.numpy()))
        Metric1 = TestMetric(Train_X, Train_Y, model, 1)#Test_X, Test_Y
        if e == 0:
            Metric = Metric1
            W = model.embed_layer.weight
        if Metric < Metric1:
            Metric = Metric1
            W = model.embed_layer.weight
    print('下一事件准确率：',Metric)


    # for i, label in zip(ConvertReflact[0].keys(),ConvertReflact[0].values()):
    #     x, y = W[i,0], W[i,1]
    #     plt.scatter(x.detach().numpy(), y.detach().numpy())
    #     plt.annotate(label, xy=(x.detach().numpy(), y.detach().numpy()), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.show()

    # X_tsne = TSNE(n_components=2, learning_rate=0.1).fit_transform(W.detach().numpy())
    # for i, label in zip(ConvertReflact[0].keys(), ConvertReflact[0].values()):
    #     if i == 0:
    #         break
    #     x, y = X_tsne[i, 0], X_tsne[i, 1]# - 1
    #     plt.scatter(x, y)
    #     plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Activity')
    # plt.show()
    return W, Metric

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

def toolGensim(Train,ConvertReflact):
    model = wv.Word2Vec(sentences=Train, vector_size=10, alpha=0.0005, window=1, min_count=1,
                        sample=0, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, compute_loss=True,
                        negative=5,  cbow_mean=1, hashfxn=hash, epochs=500, sorted_vocab=1, callbacks=[callback()])
    count = 0
    sum = 0
    for line1 in Train:
        for i in range(len(line1)-2):
            pre = model.predict_output_word([line1[i], line1[i+2]])
            if pre[0][0]==line1[i+1]:
                count+=1
            sum+=1
    print(count/sum)
    # a = model.train(Train, total_examples=1, epochs=1)
    print(model.get_latest_training_loss()/500)
    y = model.wv.similarity(1, 2)
    print(y)
    for i in model.wv.most_similar(1):
        print(i[0], i[1])

    for i, label in zip(ConvertReflact[0].keys(),ConvertReflact[0].values()):
        if i == 0:
            break
        W = model.syn1neg
        x, y = W[i-1,0], W[i-1,1]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(model.syn1neg)
    for i, label in zip(ConvertReflact[0].keys(), ConvertReflact[0].values()):
        if i == 0:
            break
        W = X_tsne
        x, y = W[i - 1, 0], W[i - 1, 1]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()
    print()