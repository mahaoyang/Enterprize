import warnings

warnings.filterwarnings("ignore")

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import keras
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

# English source data
with open("data/small_vocab_en", "r", encoding="utf-8") as f:
    source_text = f.read()

# French target data
with open("data/small_vocab_fr", "r", encoding="utf-8") as f:
    target_text = f.read()

view_sentence_range = (0, 10)

# Separate the source language text by spaces, to see how many distinct words are contained in it
print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

# 统计英文语料数据
print("-" * 5 + "English Text" + "-" * 5)
sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

# 统计法语语料数据
print()
print("-" * 5 + "French Text" + "-" * 5)
sentences = target_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

# 打印语料的前10个句子
print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

# 构造英文词典
source_vocab = list(set(source_text.lower().split()))
# 构造法语词典
target_vocab = list(set(target_text.lower().split()))

print("The size of English vocab is : {}".format(len(source_vocab)))
print("The size of French vocab is : {}".format(len(target_vocab)))

# 增加特殊编码
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

# 构造英文语料的映射表
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

# 构造法语语料的映射表
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

print("The size of English Map is : {}".format(len(source_vocab_to_int)))
print("The size of French Map is : {}".format(len(target_vocab_to_int)))


def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    Encoding the text into integers.

    @param sentence: 完整的句子，str类型
    @param map_dict: 单词到数字编码的映射
    @param max_length: 最大句子长度
    @param is_target: 当前传入的句子是否是目标语句。
                      对于目标语句，我们要在末尾添加"<EOS>"
    """

    text_to_idx = []
    # 特殊单词的数字编码
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果不是目标语句（即源语句）
    if not is_target:
        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 目标语句要对结尾添加"<EOS>"
    else:
        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 超长句子进行截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 不足长度的句子进行"<PAD>"
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx


# 对英文语料进行编码，其中设置英文句子最大长度为20
Tx = 20
source_text_to_int = []

for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, Tx, is_target=False))

# 对法语语料进行编码，其中设置法语句子最大长度为25
Ty = 25
target_text_to_int = []

for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, Ty, is_target=True))

random_index = 77

print("-" * 5 + "English example" + "-" * 5)
print(source_text.split("\n")[random_index])
print(source_text_to_int[random_index])

print()
print("-" * 5 + "French example" + "-" * 5)
print(target_text.split("\n")[random_index])
print(target_text_to_int[random_index])

from keras.utils import to_categorical

X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)

# 对X和Y做One Hot Encoding
Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(source_vocab_to_int)), X)))
Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(target_vocab_to_int)), Y)))


# 自定义softmax函数
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


# 定义全局网络层对象
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation="tanh")
densor_relu = Dense(1, activation="relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """

    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context


# 加载预训练好的glove词向量# 加载预训练好
with open("/Users/mahaoyang/PycharmProjects/Enterprize/data/glove.6B.100d.txt", 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)


def pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int):
    """
    构造Embedding层并加载预训练好的词向量（这里我使用的是100维）

    @param word_to_vec_map: 单词到向量的映射
    @param word_to_index: 单词到数字编码的映射
    """

    vocab_len = len(source_vocab_to_int) + 1  # Keras Embedding的API要求+1
    emb_dim = word_to_vec_map["the"].shape[0]

    # 初始化embedding矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # 用词向量填充embedding矩阵
    for word, index in source_vocab_to_int.items():
        word_vector = word_to_vec_map.get(word, np.zeros(emb_dim))
        emb_matrix[index, :] = word_vector

    # 定义Embedding层，并指定不需要训练该层的权重
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # build
    embedding_layer.build((None,))

    # set weights
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# 获取Embedding layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int)

n_a = 32  # The hidden size of Bi-LSTM
n_s = 128  # The hidden size of LSTM in Decoder
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(target_vocab_to_int), activation=softmax)

# 定义网络层对象（用在model函数中）
reshapor = Reshape((1, len(target_vocab_to_int)))
concator = Concatenate(axis=-1)


def model(Tx, Ty, n_a, n_s, source_vocab_size, target_vocab_size):
    """
    构造模型

    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    @param source_vocab_size: 输入（英文）语料的词典大小
    @param target_vocab_size: 输出（法语）语料的词典大小
    """

    # 定义输入层
    X = Input(shape=(Tx,))
    # Embedding层
    embed = embedding_layer(X)
    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    out0 = Input(shape=(target_vocab_size,), name='out0')
    out = reshapor(out0)

    s = s0
    c = c0

    # 模型输出列表，用来存储翻译的结果
    outputs = []

    # 定义Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(embed)

    # Decoder端，迭代Ty轮，每轮生成一个翻译结果
    for t in range(Ty):
        # 获取Context Vector
        context = one_step_attention(a, s)

        # 将Context Vector与上一轮的翻译结果进行concat
        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])

        # 将LSTM的输出结果与全连接层链接
        out = output_layer(s)

        # 存储输出结果
        outputs.append(out)

    model = Model([X, s0, c0, out0], outputs)

    return model


model = model(Tx, Ty, n_a, n_s, len(source_vocab_to_int), len(target_vocab_to_int))

model.summary()

out = model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

# 初始化各类向量# 初始化各类向
m = X.shape[0]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
out0 = np.zeros((m, len(target_vocab_to_int)))
outputs = list(Yoh.swapaxes(0, 1))

# 训练模型
model.fit([X, s0, c0, out0], outputs, epochs=5, batch_size=128)

# 保存参数
model.save_weights("pretrained_seq2seq_model.h5")

# Have a look at source text
print(source_text.split("\n")[:100])


def make_prediction(sentence):
    """
    对给定的句子进行翻译
    """
    # 将句子分词后转化为数字编码
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence.lower().split()]

    word_idx = np.array(word_idx + [0] * (20 - len(word_idx)))

    # 翻译结果
    preds = model.predict([word_idx.reshape(-1, 20), s0, c0, out0])
    predictions = np.argmax(preds, axis=-1)

    # 转换为单词
    idx = [target_int_to_vocab.get(idx[0], "<UNK>") for idx in predictions]

    # 返回句子
    return " ".join(idx)


your_sentence = input("Please input your sentences: ")

make_prediction(your_sentence)

import seaborn as sns


def plot_attention(sentence, Tx=20, Ty=25):
    """
    可视化Attention层

    @param sentence: 待翻译的句子，str类型
    @param Tx: 输入句子的长度
    @param Ty: 输出句子的长度
    """

    X = np.array(text_to_int(sentence, source_vocab_to_int))
    f = K.function(model.inputs, [model.layers[9].get_output_at(t) for t in range(Ty)])

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    out0 = np.zeros((1, len(target_vocab_to_int)))

    r = f([X.reshape(-1, 20), s0, c0, out0])

    attention_map = np.zeros((Ty, Tx))
    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0, t_prime, 0]

    Y = make_prediction(sentence)

    source_list = sentence.split()
    target_list = Y.split()

    f, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(attention_map, xticklabels=source_list, yticklabels=target_list, cmap="YlGnBu")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)


plot_attention("she likes mangoes , apples , and bananas .")

plot_attention("california is never pleasant during winter , and it is sometimes wonderful in december .")

from nltk.translate.bleu_score import sentence_bleu

# 存储每个句子的模型翻译结果# 存储每个句子
fr_preds = []

# 对样本中的每个英文进行翻译
for sentence in tqdm.tqdm(source_text.split("\n")):
    fr_pred = make_prediction(sentence)
    # 存储翻译结果
    fr_preds.append(fr_pred)

# 以样本中的法语翻译结果为reference
references = target_text.split("\n")

# 存储每个句子的BLEU分数
bleu_score = []

for i in tqdm.tqdm(range(len(fr_preds))):
    # 去掉特殊字符
    pred = fr_preds[i].replace("<EOS>", "").replace("<PAD>", "").rstrip()
    reference = references[i].lower()
    # 计算BLEU分数
    score = sentence_bleu([reference.split()], pred.split())

    bleu_score.append(score)

print("The BLEU score on our corpus is about {}".format(sum(bleu_score) / len(bleu_score)))
