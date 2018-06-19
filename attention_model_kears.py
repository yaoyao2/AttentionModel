# coding: utf-8
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np

import random
import math
import json
import os
from keras.utils.vis_utils import plot_model
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

"""
该程序是教初学者如何使用kears搭建一个自己的Attention Network。
该程序解决的任务是：将人类语言描述的时间 转换成 数字时间，自己打开Time Dataset.json文件就能很好的理解。
注意力机制是在
"""


layer1_size = 32
layer2_size = 64 # Attention layer


##################################################################
#Define some general helper methods. They are used to help tokenize data.
def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    """
    A method for tokenizing data.

    Inputs:
    dataset - A list of sentence data pairs.
    human_vocab - A dictionary of tokens (char) to id's.
    machine_vocab - A dictionary of tokens (char) to id's.
    Tx - X data size
    Ty - Y data size

    Outputs:
    X - Sparse tokens for X data
    Y - Sparse tokens for Y data
    Xoh - One hot tokens for X data
    Yoh - One hot tokens for Y data
    """

    # Metadata
    m = len(dataset)

    # Initialize
    X = np.zeros([m, Tx], dtype='int32')
    Y = np.zeros([m, Ty], dtype='int32')

    # Process data
    for i in range(m):
        data = dataset[i]
        X[i] = np.array(tokenize(data[0], human_vocab, Tx))
        Y[i] = np.array(tokenize(data[1], machine_vocab, Ty))

    # Expand one hots
    Xoh = oh_2d(X, len(human_vocab))
    Yoh = oh_2d(Y, len(machine_vocab))

    return (X, Y, Xoh, Yoh)


def tokenize(sentence, vocab, length):
    """
    Returns a series of id's for a given input token sequence.

    It is advised that the vocab supports <pad> and <unk>.

    Inputs:
    sentence - Series of tokens
    vocab - A dictionary from token to id
    length - Max number of tokens to consider

    Outputs:
    tokens -
    """
    tokens = [0] * length
    for i in range(length):
        char = sentence[i] if i < len(sentence) else "<pad>"
        char = char if (char in vocab) else "<unk>"
        tokens[i] = vocab[char]

    return tokens


def ids_to_keys(sentence, vocab):
    """
    Converts a series of id's into the keys of a dictionary.
    """
    return [list(vocab.keys())[id] for id in sentence]


def oh_2d(dense, max_value):
    """
    Create a one hot array for the 2D input dense array.
    """
    # Initialize
    oh = np.zeros(np.append(dense.shape, [max_value]))

    # Set correct indices
    ids1, ids2 = np.meshgrid(np.arange(dense.shape[0]), np.arange(dense.shape[1]))

    oh[ids1.flatten(), ids2.flatten(), dense.flatten('F').astype(int)] = 1

    return oh


########################################################################




if  __name__=='__main__':

    ####1、先加载一些数据
    with open('data/Time Dataset.json', 'r') as f:
        #加载训练数据集  是一些转换对  [人类语言描述的时间，数字时间]
        dataset = json.loads(f.read(),encoding='utf-8')
    with open('data/Time Vocabs.json', 'r') as f:
        #加载人类语言描述时间会用到的所有字符human_vocab
        #加载数字时间会用到的所有字符machine_vocab
        human_vocab, machine_vocab = json.loads(f.read(),encoding='utf-8')

    human_vocab_size = len(human_vocab)#41个字符
    machine_vocab_size = len(machine_vocab)#11个字符

    ####Number of training examples
    m = len(dataset) #训练集有10000个实例

    # print(human_vocab)
    # print(machine_vocab)
    # print('human_vocab_size',human_vocab_size)
    # print('machine_vocab_size',machine_vocab_size)
    # print('Number of training examples',m)

    ####2、对数据进行切分处理，希望得到其由单个字符表示的形式
    ####eg：['t11:36', '11:36']-----> [['t','1','1',':','3','6'],['1','1',':','3','6']]
    Tx = 41  # x序列的最大长度，自己根据数据可以找到最大长度
    Ty = 5  # y因为是数字时间，所有长度都一样，所有y序列的长度为5  eg：20:34
    #X,Y是分别根据human_vocab字典和machine_vocab字典，生成的字符索引序列
    #Xoh，Yoh是one-hot向量
    #Xoh的维度是41*41   维度1：x序列的最大长度  维度2：human_vocab字典的长度
    #Yoh的维度是5*11    维度1：y序列的长度   维度2：machine_vocab字典的长度
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    # Split data 80-20 between training and test划分训练集和测试集
    train_size = int(0.8 * m)
    Xoh_train = Xoh[:train_size]
    Yoh_train = Yoh[:train_size]
    Xoh_test = Xoh[train_size:]
    Yoh_test = Yoh[train_size:]

    ######检查##########
    i = 0
    print("Input data point " + str(i) + ".")
    print("")
    print("The data input is: " + str(dataset[i][0]))
    print("The data output is: " + str(dataset[i][1]))
    print("")
    print("The tokenized input is:" + str(X[i]))
    print("The tokenized output is: " + str(Y[i]))
    print("")
    print("The one-hot input is:", Xoh[i])#41*41
    print("The one-hot output is:", Yoh[i])#5*11
    print("")
    print("The one-hot input shape is:",Xoh[i].shape)
    print("The one-hot output shape is:", Yoh[i].shape)
    ######检查##########



    ####3、定义模型
####************************定义模型需要的函数**************************####
# Define part of the attention layer gloablly so as to
# share the same layers for each attention step.
def softmax(x):
    # return K.softmax(x, axis=1)
    return tf.nn.softmax(x, axis=1)

###此处是设置一些默认的参数
at_repeat = RepeatVector(Tx,name='at_repeat__one_step_of_attention')  #Tx=41  RepeatVector()对向量进行重复
at_concatenate = Concatenate(axis=-1,name='at_concatenate__one_step_of_attention') #Concatenate接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。
at_dense1 = Dense(8, activation="tanh",name='at_dense1_tanh__one_step_of_attention') #全连接层
at_dense2 = Dense(1, activation="relu",name='at_dense2_relu__one_step_of_attention') #全连接层
at_softmax = Activation(softmax, name='attention_weights__one_step_of_attention')#获取注意力权重
at_dot = Dot(axes=1,name='at_dot__one_step_of_attention') #计算乘积


def one_step_of_attention(h_prev, a):#简称
    # h是Lambda层处理后的中间语义; X是a1,是Encoder得到的中间语义
    # h_prev=h; a=X
    """
    Get the context.

    Input:
    h_prev - Previous hidden state of a RNN layer (m, n_h)
    a - Input data, possibly processed (m, Tx, n_a)

    Output:
    context - Current context (m, Tx, n_a)
    """
    # Repeat vector to match a's dimensions
    # 对向量h_prev进行重复,会重复Tx次,这样就能匹配上a的维度
    h_repeat = at_repeat(h_prev)

    # Calculate attention weights
    # 计算注意力权重
    i = at_concatenate([a, h_repeat])
    i = at_dense1(i)
    i = at_dense2(i)
    attention = at_softmax(i)
    # Calculate the context
    #得到文本的注意力权重
    context = at_dot([attention, a])

    return context


def attention_layer(X, n_h, Ty):
    # a1是Encoder得到的中间语义,layer2_size=64,Ty=5
    # X=a1;n_h=layer2_size=64;Ty=5
    """
    Creates an attention layer.

    Input:
    X - Layer input (m, Tx, x_vocab_size)
    n_h - Size of LSTM hidden layer
    Ty - Timesteps in output sequence

    Output:
    output - The output of the attention layer (m, Tx, n_h)
    """
    # Define the default state for the LSTM layer
    #将Encoder得到的中间语义X数据进行变换,变换本身没有什么需要学习的参数,所以使用Lambda层
    #X:mx41x64 --> h:mx64
    #X:mx41x64 --> c:mx64
    h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)),name='h__attention_layer')(X)
    c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)),name='c__attention_layer')(X)
    # Messy, but the alternative is using more Input()

    at_LSTM = LSTM(n_h, return_state=True,name='at_LSTM__attention_layer')

    output = []

    # Run attention step and RNN for each output time step
    #此处是动态的生成Ci的过程,它是对生成目标Y是有顺序要求的
    for _ in range(Ty):
        #注意力Ci的生成过程函数one_step_of_attention()
        #h是Lambda层处理后的中间语义; X是a1,是Encoder得到的中间语义
        context = one_step_of_attention(h, X)


        #context是通过注意力模型得到的每个文本的注意力权值计算结果
        #输入初始化的h和c,然后再不断的改变h,c
        h, _, c = at_LSTM(context, initial_state=[h, c])

        output.append(h)

    return output
####************************定义模型需要的函数**************************####


#解码器Decoder
layer3 = Dense(machine_vocab_size, activation=softmax, name='layer3_Decoder')

def get_model(Tx, Ty, layer1_size, layer2_size, x_vocab_size, y_vocab_size):
    """
    Creates a model.

    input:
    Tx - Number of x timesteps                                       Tx是：x序列的长度41
    Ty - Number of y timesteps                                       Ty是：y序列的长度5
    size_layer1 - Number of neurons in BiLSTM                        size_layer1是：BiLSTM的神经元个数32
    size_layer2 - Number of neurons in attention LSTM hidden layer   size_layer2是：注意力LSTM隐藏层的神经元个数64
    x_vocab_size - Number of possible token types for x              x_vocab_size是：41
    y_vocab_size - Number of possible token types for y              y_vocab_size是：5

    Output:
    model - A Keras Model.
    """

    # Create layers one by one在搭建模型
    #Tx=41,x序列的最大长度; x_vocab_size=41,human_vocab_size人类表述时间会用到41个不同的字符
    #输入一个x实例 (41x41的矩阵),输出41x41
    X = Input(shape=(Tx, x_vocab_size),name='X__get_model')

    #构建了一个双向LSTM模型,layer1_size=32,输出41x64
    #此处可以相当于Encoder层,使用的方法是双向LSTM对原始数据进行编码
    a1 = Bidirectional(LSTM(layer1_size, return_sequences=True), merge_mode='concat',name='a1_Encoder__get_model')(X)

    #自定义的注意力层
    #a1是Encoder得到的中间语义,layer2_size=64,Ty=5
    #使用注意力模型对a1进行了处理
    a2 = attention_layer(a1, layer2_size, Ty)

    #将a2中每一步得到的计算值,输入到layer3这个解码器中,这个解码器就是一个全连接层
    a3 = [layer3(timestep) for timestep in a2]

    # Create Keras model
    model = Model(inputs=[X], outputs=a3)

    return model






# Obtain a model instance获取模型的实例
"""
# Tx=41 x序列的最大长度
# Ty=5 y序列的长度
# layer1_size=32
# layer2_size=64 #注意力层
# human_vocab_size=41  人类表述时间会用到41个不同的字符
# machine_vocab_size=11 数字表述时间会用到11个不同的字符
"""
model = get_model(Tx, Ty, layer1_size, layer2_size, human_vocab_size, machine_vocab_size)

###save model
# model.save('AttentionModel.h5')
###打印模型图片
plot_model(model,to_file='AttentionModel_new.png', show_shapes=True)



# Create optimizer
opt = Adam(lr=0.05, decay=0.04, clipnorm=1.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Group the output by timestep, not example
outputs_train = list(Yoh_train.swapaxes(0,1))


# Time to train
# It takes a few minutes on an quad-core CPU
model.fit([Xoh_train], outputs_train, epochs=30, batch_size=100)









# #####################################模型评估#################################
# # Evaluate the test performance
# outputs_test = list(Yoh_test.swapaxes(0,1))
# score = model.evaluate(Xoh_test, outputs_test)
# print('Test loss: ', score[0])
#
#
#
# # Let's visually check model output.
# import random as random
#
# i = random.randint(0, m)
#
# def get_prediction(model, x):
#     prediction = model.predict(x)
#     max_prediction = [y.argmax() for y in prediction]
#     str_prediction = "".join(ids_to_keys(max_prediction, machine_vocab))
#     return (max_prediction, str_prediction)
#
# max_prediction, str_prediction = get_prediction(model, Xoh[i:i+1])
#
# print("Input: " + str(dataset[i][0]))
# print("Tokenized: " + str(X[i]))
# print("Prediction: " + str(max_prediction))
# print("Prediction text: " + str(str_prediction))
#
# i = random.randint(0, m)
#
#
# def plot_attention_graph(model, x, Tx, Ty, human_vocab, layer=7):
#     # Process input
#     tokens = np.array([tokenize(x, human_vocab, Tx)])
#     tokens_oh = oh_2d(tokens, len(human_vocab))
#
#     # Monitor model layer
#     layer = model.layers[layer]
#
#     layer_over_time = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
#     layer_output = layer_over_time([tokens_oh])
#     layer_output = [row.flatten().tolist() for row in layer_output]
#
#     # Get model output
#     prediction = get_prediction(model, tokens_oh)[1]
#
#     # Graph the data
#     fig = plt.figure()
#     fig.set_figwidth(20)
#     fig.set_figheight(1.8)
#     ax = fig.add_subplot(111)
#
#     plt.title("Attention Values per Timestep")
#
#     plt.rc('figure')
#     cax = plt.imshow(layer_output, vmin=0, vmax=1)
#     fig.colorbar(cax)
#
#     plt.xlabel("Input")
#     ax.set_xticks(range(Tx))
#     ax.set_xticklabels(x)
#
#     plt.ylabel("Output")
#     ax.set_yticks(range(Ty))
#     ax.set_yticklabels(prediction)
#
#     plt.show()
#
#
# plot_attention_graph(model, dataset[i][0], Tx, Ty, human_vocab)
#
#










