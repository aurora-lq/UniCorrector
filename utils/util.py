import re
import torch
import unicodedata
from config.config import constant


def flit_back(s):  
    b = re.sub('_', r'\\', s)
    return b


def normalize_string(s):
    # lowercase, trim and remove non-letter characters
    s = unicode_to_ascii(s.strip())
    s = re.sub(r'\\', '_', s)  
    return s


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def filter_pair(pair):
    return (len(pair[0]) <= constant.MAX_LENGTH and len(pair[1]) <= constant.MAX_LENGTH) and (
            len(pair[0]) > constant.MIN_LENGTH and len(pair[1]) > constant.MIN_LENGTH)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def indexes_from_sentence(language, sentence):
    return [language.word2index[word] for word in sentence]


def tensor_from_sentence(language, sentence, flag):  # flag = 0 means in_en_input, 1 means de_de_input, 2 means target
    indexes = indexes_from_sentence(language, sentence)
    if flag == 0:
        for i in range(constant.MAX_LENGTH + 1 - len(indexes)):
            indexes.append(constant.PAD_TOKEN)
    elif flag == 1:
        indexes.insert(0, constant.SOS_TOKEN)
        for i in range(constant.MAX_LENGTH - len(indexes)):
            indexes.append(constant.PAD_TOKEN)
    else:
        # indexes.append(EOS_token)
        for i in range(constant.MAX_LENGTH - len(indexes)):
            indexes.append(constant.PAD_TOKEN)
        indexes.append(constant.EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=constant.DEVICE).view(-1, 1)


def split_sentence(sentence, flag):
    sentence_list = list(sentence)
    if flag == 3 and len(sentence_list) > constant.MAX_PROTEIN_LENGTH:
        sentence_list = sentence_list[:constant.MAX_PROTEIN_LENGTH]
    length = len(sentence_list)
    t = " ".join(sentence_list)
    if flag != 3:
        if flag == 0:  # en_input
            pass
        elif flag == 1:  # de_input
            t = 'SOS ' + t
            length += 1
        elif flag == 2:  # target
            t = t + ' EOS'
            length += 1
        for i in range(constant.MAX_LENGTH + 1 - length):
            t = t + ' PAD'
    else:
        if length > constant.MAX_PROTEIN_LENGTH:
            t = t[:constant.MAX_PROTEIN_LENGTH]
        for i in range(constant.MAX_PROTEIN_LENGTH + 1 - length):
            t = t + ' PAD'
    return t
