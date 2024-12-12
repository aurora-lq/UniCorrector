# coding = utf-8

from .basic_transformer import Encoder,Decoder
from .parameters import p
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import constant

d_model, d_ff, d_k, d_v, n_layers, n_heads = p.d_model, p.d_ff, p.d_k, p.d_v, p.n_layers, p.n_heads
device = constant.DEVICE


class TransTrans(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, tgt_vocab):
        super(TransTrans, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.encoder_frag = Encoder(src_vocab_size).to(device)
        self.encoder_correct = Encoder(src_vocab_size).to(device)
        self.decoder_frag = Decoder(tgt_vocab_size).to(device)
        self.decoder_correct = Decoder(tgt_vocab_size).to(device)
        self.projection_frag = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
        self.projection_correct = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
       
        enc_outputs, enc_self_attns = self.encoder_frag(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder_frag(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection_frag(dec_outputs)
        g = F.gumbel_softmax(dec_logits, 0.1, True)
        max_indices = torch.argmax(g, dim=2)
        outputs_frag = dec_logits.view(-1, dec_logits.size(-1))
        enc_correct_outputs, _ = self.encoder_correct(max_indices)
        dec_correct_outputs, _, _ = self.decoder_correct(dec_inputs, max_indices, enc_correct_outputs)
        correct_logits = self.projection_correct(dec_correct_outputs)
        outputs_correct = correct_logits.view(-1, correct_logits.size(-1))
        return outputs_frag, outputs_correct, enc_self_attns, dec_self_attns, dec_enc_attns
