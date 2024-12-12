# coding = utf-8
import random

from sklearn.model_selection import train_test_split

from utils.util import flit_back, filter_pairs, normalize_string, split_sentence
from utils.data_loader import MyDataSet, make_data
from config.config import constant
from model.basic_transformer import Transformer as Basic_Transformer
from model.unified_transformer import Transformer as Unified_Transformer
from train import TrainIterable, evaluate
import torch.utils.data as data
import torch
import torch.optim as optim
import ast
print(dir(constant))


dataset_path = ''

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

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_pre_vocab(inp, output, pro):
    return ast.literal_eval(inp), ast.literal_eval(output), ast.literal_eval(pro)

def read_dataset(language1, language2, with_protein=False, reverse=False):
    print('Reading lines...')
    lines = open(dataset_path, encoding='utf-8').read().strip().split('\n')
    print('lines第一行是：', lines[0])

    l_pairs = [[normalize_string(s) for s in l.split()] for l in lines]#这里也有修改，就是改的\t
    print('l_pairs第一行是：', l_pairs[0])

    protein = []
    [protein.append(pair[2]) for pair in l_pairs if with_protein]
    if reverse:
        if constant.INFERENCE:
            k = 3
            l_pairs = [[p[1], p[0], p[2], k] for p in l_pairs]
        else:
            l_pairs = [[p[1], p[0], p[2], float(p[4])] for p in l_pairs]  #correct_smiles	wrong_smiles	padding	qed	logp	sascore
        input_l = Seq(language2)
        output_l = Seq(language1)
    else:

        input_l = Seq(language1)
        output_l = Seq(language2)
    protein_l = Seq('pro')
    print('重新选择对应的属性后面的l_pairs:', l_pairs[0])
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

    src, tgt, pro = load_pre_vocab(
        "{0: 'PAD', 1: 'SOS', 2: 'EOS', 3: '>', 4: 'C', 5: 'O', 6: 'n', 7: '1', 8: 'c', 9: '(', 10: '-', 11: '2', 12: '#', 13: 'N', 14: ')', 15: '=', 16: '<', 17: 'F', 18: '&', 19: 'S', 20: 'o', 21: '[', 22: '@', 23: 'H', 24: ']', 25: '3', 26: '+', 27: 'l', 28: '4', 29: '/', 30: 's', 31: ';', 32: '5', 33: '6', 34: '_', 35: '?', 36: ':', 37: 'I', 38: 'B', 39: 'r', 40: 'i'}",
        "{0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'C', 4: 'O', 5: 'n', 6: '1', 7: 'c', 8: '(', 9: '=', 10: ')', 11: '-', 12: '#', 13: 'N', 14: '/', 15: '2', 16: '3', 17: 'S', 18: '[', 19: '@', 20: 'H', 21: ']', 22: 'F', 23: 's', 24: '4', 25: 'l', 26: '_', 27: 'o', 28: '5', 29: '+', 30: 'P', 31: '6', 32: 'B', 33: 'r', 34: 'I', 35: '7', 36: 'i'}",
        "{0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'M'}")
    if constant.INFERENCE:
        input_language.load(src)
        output_language.load(tgt)
        protein_language.load(pro)
    src_vocab_size = input_language.n_words
    tgt_vocab_size = output_language.n_words
    pro_vocab_size = protein_language.n_words
    
    
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
                                pro_vocab_size) if constant.WITH_PROTEIN else Basic_Transformer(
        src_vocab_size, tgt_vocab_size)

    optimizer = optim.SGD(model.parameters(), lr=constant.LEARNING_RATE, momentum=0.99)

    if constant.START_EPOCH != 0 and constant.START_EPOCH != constant.EPOCHS:

        model.load_state_dict(torch.load('{}logpepoch100checkpoint_model{}.pth'.format(constant.MODEL_PATH,constant.DEVICE)))
        optimizer.load_state_dict(torch.load('{}logpepoch100checkpoint_optimizer{}'.format(constant.MODEL_PATH,constant.DEVICE)))

    train_iter = TrainIterable(constant.EPOCHS, constant.LEARNING_RATE, model, optimizer, pairs, src_vocab, pro_vocab,
                               tgt_vocab, reverse_tgt_vocab)
    for train_i in train_iter:
        with open('{}process.txt'.format(constant.RESULT_PATH), 'a+') as f:
            f.write(str(train_i.epoch) + ' ' + str(train_i.loss))
            torch.save(train_i.model.state_dict(), '{}logpepoch100checkpoint_model{}.pth'.format(constant.MODEL_PATH, constant.DEVICE))
            torch.save(train_i.optimizer.state_dict(), '{}logpepoch100checkpoint_optimizer{}'.format(constant.MODEL_PATH, constant.DEVICE))
            f.write(' {}s\n'.format(str(train_i.time_cost)))



    model = train_iter.model

    if constant.START_EPOCH == constant.EPOCHS:
        model.load_state_dict(torch.load('{}logpepoch100model{}.pth'.format(constant.MODEL_PATH, "cuda:0"),map_location=constant.DEVICE))
    else:
        torch.save(model.state_dict(), '{}logpepoch100model{}.pth'.format(constant.MODEL_PATH, constant.DEVICE))
       
    if constant.INFERENCE:
        evaluate(pairs, model, src_vocab, pro_vocab, tgt_vocab, reverse_src_vocab, reverse_tgt_vocab, 1, 1,
                 'inference.txt', 0.99)
    else:
        evaluate(pairs, model, src_vocab, pro_vocab, tgt_vocab, reverse_src_vocab, reverse_tgt_vocab, 1, 1,
                 'TransformerResults0302.txt')
