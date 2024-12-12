from model.trans_trans import TransTrans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import flit_back
import re
from config.config import constant

device = constant.DEVICE

d = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'C', 4: '(', 5: ')', 6: '=', 7: 'O', 8: 'N', 9: '1', 10: '[', 11: '@', 12: 'H',
     13: ']', 14: '2', 15: 'c', 16: '3', 17: 'n', 18: 's', 19: 'F', 20: '/', 21: '_', 22: '#', 23: 'o', 24: 'S',
     25: '4', 26: 'l', 27: '-', 28: 'B', 29: 'r', 30: '+', 31: '5', 32: 'I', 33: 'i', 34: '6', 35: '7', 36: 'P'}

rd = {v: k for k, v in d.items()}


def inference(smiles, vocab):
    trans = TransTrans(38, 37, rd)
    trans.load_state_dict(torch.load('./results/checkpoint_model2.pth'))
    trans.to(device)
    Encoder = trans.encoder_correct
    Decoder = trans.decoder_correct
    Project = trans.projection_correct
    smiles_tensor = [rd[s] for s in smiles]
    for i in range(constant.MAX_LENGTH - len(smiles_tensor)):
        smiles_tensor.append(0)
    enc_inputs = torch.tensor(smiles_tensor).unsqueeze(0).to(device)
    print(enc_inputs.shape)
    enc_correct_outputs, _ = Encoder(enc_inputs)
    dec_inputs = torch.full((enc_inputs.size(0), 1), constant.SOS_TOKEN, dtype=enc_inputs.dtype).to(device)
    terminal = torch.zeros(enc_inputs.size(0), dtype=torch.bool).to(device)

    for _ in range(constant.MAX_LENGTH):
        dec_outputs, _, _ = Decoder(dec_inputs, enc_inputs, enc_correct_outputs)

        projected = Project(dec_outputs)
        next_symbols = projected[:, -1, :].max(dim=-1)[1]
        dec_inputs = torch.cat([dec_inputs, next_symbols.unsqueeze(1)], dim=1)

      
        terminal |= next_symbols == constant.EOS_TOKEN
        if terminal.all():
            break
    print(dec_inputs)
    with open('{}{}'.format(constant.RESULT_PATH, "transinference_gpt.txt"), 'a+') as f:
        for i in range(enc_inputs.size(0)):
            res = flit_back(
                re.sub('EOS.*', '', (''.join([d[t.item()] for t in dec_inputs[i, 1:]]))))
            print(res)
            f.writelines("C" + '\t' + smiles + '\t' + res + '\n')


with open("./datasets/dataset_gpt_generated.csv", 'r+') as f:
    lines = f.readlines()
    for line in lines:
        smiles = line.split('\t')[1] 
        inference(smiles,rd)
