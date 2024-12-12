# coding = utf-8
import torch
import torch.utils.data as Data


class MyDataSet(Data.Dataset):
 

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# def make_data(sentences, src_vocab, tgt_vocab):
#   
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#     for i in range(len(sentences)):
#         enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
#         dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
#         dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
#         enc_inputs.extend(enc_input)
#         dec_inputs.extend(dec_input)
#         dec_outputs.extend(dec_output)
#
#     return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataSetPro(Data.Dataset):
    def __init__(self, enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs):
        super(MyDataSetPro, self).__init__()
        self.enc_mol_inputs = enc_mol_inputs
        self.enc_pro_inputs = enc_pro_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_mol_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_mol_inputs[idx], self.enc_pro_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class MyDataSetLazy(Data.Dataset):
    def __init__(self, sentences, src_vocab, pro_vocab, tgt_vocab):
        super(MyDataSetLazy, self).__init__()
        self.sentences = sentences
        self.src_vocab = src_vocab
        self.pro_vocab = pro_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.sentences)
# ['< C [ C @ ] 1 ( C ( = O ) N c 2 c n c ( C ( = O ) O ) s 2 ) C [ C @ @ H ] 2 C [ C @ @ H ] 2 F PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD', 'M PAD', 'SOS C C O C ( = O ) c 1 n c c ( N C ( = O ) [ C @ @ ] 2 ( C ) C [ C @ @ H ] 2 F ) s 1 PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD', 'C C O C ( = O ) c 1 n c c ( N C ( = O ) [ C @ @ ] 2 ( C ) C [ C @ @ H ] 2 F ) s 1 EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD', '0.8512171115055513']
    def __getitem__(self, idx):
        enc_mol_input = [self.src_vocab[n] for n in self.sentences[idx][0].split()]
        enc_pro_input = [self.pro_vocab[n] for n in self.sentences[idx][1].split()]
        dec_input = [self.tgt_vocab[n] for n in self.sentences[idx][2].split()]
        dec_output = [self.tgt_vocab[n] for n in self.sentences[idx][3].split()]
        #enc_prop = [self.sentences[idx][4]]
        enc_prop = float(self.sentences[idx][4])
        return torch.LongTensor(enc_mol_input), torch.LongTensor(enc_pro_input), torch.tensor([enc_prop], dtype = torch.float), torch.LongTensor(
            dec_input), torch.LongTensor(dec_output),
        # return self.enc_mol_inputs[idx], self.enc_pro_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


def make_data(sentences, src_vocab, pro_vocab, tgt_vocab):
    enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs = [], [], [], []
    for i in range(len(sentences)):
        enc_mol_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        enc_pro_input = [[pro_vocab[n] for n in sentences[i][1].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][3].split()]]
        enc_mol_inputs.extend(enc_mol_input)
        enc_pro_inputs.extend(enc_pro_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_mol_inputs), torch.LongTensor(enc_pro_inputs), torch.LongTensor(
        dec_inputs), torch.LongTensor(dec_outputs)

