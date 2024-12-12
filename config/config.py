# coding = utf-8
import sys

from . import constant
import torch


constant.MAX_LENGTH = 110
constant.MIN_LENGTH = 0
constant.MAX_PROTEIN_LENGTH = 1
constant.START_EPOCH = 1
constant.EPOCHS =1 #100
constant.BATCH_SIZE = 128
constant.LEARNING_RATE = 1e-3
constant.QED = False
constant.MOL = not constant.QED
constant.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
constant.PAD_TOKEN = 0
constant.SOS_TOKEN = 1
constant.EOS_TOKEN = 2
constant.TRAINING_PAIRS_SEED = 1
constant.TEST_SIZE = 0.97
constant.RESULT_PATH = './results/'
constant.MODEL_PATH = './results/'
constant.WITH_PROTEIN = False
constant.INFERENCE = True 
#constant.props = 'qed'
