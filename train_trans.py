import random

from sklearn.model_selection import train_test_split

from utils.util import flit_back, filter_pairs, normalize_string, split_sentence
from utils.data_loader import MyDataSet, make_data
from config.config import constant
from model.basic_transformer import Transformer as Basic_Transformer
from model.unified_transformer import Transformer as Unified_Transformer
from model.trans_trans import TransTrans
from train_TransTrans import TrainIterable, evaluate
import torch.utils.data as data
import torch
import torch.optim as optim
import ast

dataset_path = "../dataset/zinc_2kw.txt"


class Seq:
    def __init__(self, name):
        self.name = name
        self.word2index = {'PAD': 0, 'SOS': 1, 'EOS': 2}  # vocabulary
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'SOS', 2: "EOS"}
        self.n_words = 3  # PAD, SOS, EOS

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def load(self, index2word_pre):
        self.index2word = index2word_pre
        self.word2index = {value: key for key, value in index2word_pre.items()}
        self.n_words = len(self.word2index)


def load_pre_vocab(inp, output, pro):
    return ast.literal_eval(inp), ast.literal_eval(output), ast.literal_eval(pro)


def read_dataset(language1, language2, with_protein=False, reverse=False):
    print('Reading lines...')
    lines = open(dataset_path, encoding='utf-8').read().strip().split('\n')
    l_pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    protein = []
    [protein.append(pair[2]) for pair in l_pairs if with_protein]
    if reverse:
        l_pairs = [[p[1], p[0], p[2]] for p in l_pairs]
        input_l = Seq(language2)
        output_l = Seq(language1)
    else:
        input_l = Seq(language1)
        output_l = Seq(language2)
    protein_l = Seq('pro')
    return input_l, output_l, protein_l, l_pairs


def prepare_data(language1, language2, reverse=False, with_protein=False):
    input_l, output_l, protein_l, l_pairs = read_dataset(language1, language2, with_protein, reverse)
    print('Read {} pairs'.format(len(l_pairs)))
    print(l_pairs[0])
    l_pairs = filter_pairs(l_pairs)
    print('Trimmed to {} pairs'.format(len(l_pairs)))
    for pair in l_pairs:
        input_l.add_sentence(pair[0])
        output_l.add_sentence(pair[1])
        if len(pair) > 2:
            protein_l.add_sentence(pair[2])
    print(input_l.name, input_l.n_words)
    print(output_l.name, output_l.n_words)
    print(protein_l.name, protein_l.n_words)
    return input_l, output_l, protein_l, l_pairs


if __name__ == '__main__':
    input_language, output_language, protein_language, pairs = prepare_data('output', 'input', True,
                                                                            constant.WITH_PROTEIN)
    src_vocab_size = input_language.n_words
    tgt_vocab_size = output_language.n_words
    pro_vocab_size = protein_language.n_words
    print(src_vocab_size)
    print(tgt_vocab_size)
    print(pro_vocab_size)
    src_vocab, tgt_vocab, pro_vocab = input_language.word2index, output_language.word2index, protein_language.word2index
    reverse_src_vocab, reverse_tgt_vocab = input_language.index2word, output_language.index2word
    print(random.choice(pairs))
    print(input_language.index2word)
    print(output_language.index2word)
    print(protein_language.index2word)
    print("OK! We have processed the data successfully!")
    model = Unified_Transformer(src_vocab_size, tgt_vocab_size,
                                pro_vocab_size) if constant.WITH_PROTEIN else TransTrans(
        src_vocab_size, tgt_vocab_size, reverse_tgt_vocab)
    optimizer = optim.SGD(model.parameters(), lr=constant.LEARNING_RATE, momentum=0.99)
    if constant.START_EPOCH != 0:
        model.load_state_dict(torch.load('{}checkpoint_model2.pth'.format(constant.MODEL_PATH)))
        optimizer.load_state_dict(torch.load('{}checkpoint_optimizer2'.format(constant.MODEL_PATH)))

    train_iter = TrainIterable(constant.EPOCHS, constant.LEARNING_RATE, model, optimizer, pairs, src_vocab, pro_vocab,
                               tgt_vocab, reverse_tgt_vocab)
    for train_i in train_iter:
        with open('{}process.txt'.format(constant.RESULT_PATH), 'a+') as f:
            f.write(str(train_i.epoch) + ' ' + str(train_i.loss))
            torch.save(train_i.model.state_dict(), '{}checkpoint_model2.pth'.format(constant.MODEL_PATH))
            torch.save(train_i.optimizer.state_dict(), '{}checkpoint_optimizer2'.format(constant.MODEL_PATH))
            f.write(' {}s\n'.format(str(train_i.time_cost)))
        # continue
    model = train_iter.model
    torch.save(model.state_dict(), '{}model2.pth'.format(constant.MODEL_PATH))
    model.load_state_dict(torch.load('{}model2.pth'.format(constant.MODEL_PATH)))
    if constant.INFERENCE:
        evaluate(pairs, model, src_vocab, pro_vocab, tgt_vocab, reverse_src_vocab, reverse_tgt_vocab, 1, 1,
                 'inference.txt', 1)
    else:
        evaluate(pairs, model, src_vocab, pro_vocab, tgt_vocab, reverse_src_vocab, reverse_tgt_vocab, 1, 1,
                 'TransformerResults.txt')
   
