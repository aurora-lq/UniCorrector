from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import time
import re
from sklearn.model_selection import train_test_split
from utils.util import flit_back
from rdkit import Chem
from rdkit.Chem import QED
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from config.config import constant
from utils.util import split_sentence
from utils.data_loader import MyDataSetLazy, MyDataSetPro, make_data
import seaborn
from tqdm import tqdm

device = constant.DEVICE


def make_dataset(pairs, src_vocab, pro_vocab, tgt_vocab):
    train_pairs, _ = train_test_split(pairs, test_size=constant.TEST_SIZE,
                                      random_state=constant.TRAINING_PAIRS_SEED)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=constant.TEST_SIZE,
                                              random_state=constant.TRAINING_PAIRS_SEED)
    train_loader, val_loader = pairs_to_loader(train_pairs, src_vocab, pro_vocab, tgt_vocab), pairs_to_loader(val_pairs,
                                                                                                              src_vocab,
                                                                                                              pro_vocab,
                                                                                                              tgt_vocab)
    return train_loader, val_loader


def pairs_to_loader(pairs, src_vocab, pro_vocab, tgt_vocab):
    sentences = []
    for pair in pairs:
        sentences.append([split_sentence(pair[0], 0), split_sentence(pair[2], 3), split_sentence(pair[1], 1),
                          split_sentence(pair[1], 2)])
    print(sentences[0])
    # enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, pro_vocab,
    #                                                                     tgt_vocab)
    # return Data.DataLoader(MyDataSetPro(enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs),
    #                        constant.BATCH_SIZE, True)
    return Data.DataLoader(MyDataSetLazy(sentences, src_vocab, pro_vocab, tgt_vocab), constant.BATCH_SIZE, True,
                           num_workers=6)


class TrainIterable:
    def __init__(self, epochs, learning_rate, model, optimizer, pairs, src_vocab, pro_vocab, tgt_vocab,
                 reverse_tgt_vocab):
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._model = model
        self._pairs = pairs
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._pro_vocab = pro_vocab
        self._reverse_tgt_vocab = reverse_tgt_vocab
        self._epoch = constant.START_EPOCH
        self._loss = []
        self._train_plt_loss = []
        self._validate_plt_loss = []
        self._optimizer = optimizer
        self._train_loader, self._val_loader = make_dataset(pairs, src_vocab, pro_vocab, tgt_vocab)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def epoch(self):
        return self._epoch

    @property
    def time_cost(self):
        return self._time_cost

    @property
    def loss(self):
        return self._loss

    @property
    def train_plt_loss(self):
        return self._train_plt_loss

    def __iter__(self):
        return self

    def __next__(self):
        if self._epoch < self._epochs:
            time_start = time.time()
            self._model, self._optimizer, train_avg_loss, validate_loss, self.enc_self_attns, self.dec_self_attns, self.dec_enc_attns = train(
                self._train_loader, self._val_loader,
                self._optimizer, self._model,
                device, self._reverse_tgt_vocab)
            time_end = time.time()
            print('Epoch:', '%04d' % (self._epoch + 1), 'train_loss =', '{:.6f}'.format(train_avg_loss), 'val_loss',
                  '{:.6f}'.format(validate_loss))
            print('time cost', (time_end - time_start), 's')
            self._time_cost = time_end - time_start
            self._train_plt_loss.append(train_avg_loss)
            self._validate_plt_loss.append(validate_loss)
            self._loss = ('{:.6f}'.format(train_avg_loss), '{:.6f}'.format(validate_loss))
            self._epoch += 1
            return self
        else:
            self.plot()
            # self.plot_attn()
            raise StopIteration

    def plot(self):
        x = np.arange(1 + constant.START_EPOCH, self._epoch + 1, 1)
        plt.plot(x, self._train_plt_loss)
        plt.plot(x, self._validate_plt_loss)
        plt.xlabel(u"epoch")
        plt.ylabel(u"loss")
        plt.legend(['train', 'val'], loc='best')
        plt.savefig('./results/train_loss.jpg')

    def plot_heat_map(self, data, x, y, ax):
        seaborn.heatmap(data.detach().cpu().numpy(),
                        xticklabels=x, square=True, yticklabels=y, fmt='.2f', annot=True, ax=ax, cmap='Greens',
                        annot_kws={'size': 20})

    def plot_attn(self):
        *_, dec_enc_attns, mol_fragment, mol = translate_one(model=self.model, src_vocab=self._src_vocab,
                                                             tgt_vocab=self._tgt_vocab,
                                                             pro_vocab=self._pro_vocab, pro="M")
        fig, axs = plt.subplots(1, 1, figsize=(80, 80))  
        axs.set_xticklabels(axs.get_xticklabels(), fontsize=30)
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=30)
        # fig,axs = plt.subplots(figsize=(15,10))
        sum_dec_enc_attn = torch.sum(dec_enc_attns[0][0], 0)
        # for h in range(1):  # 每一个循环都是一个头的注意力
        #     self.plot_heat_map(
        #         dec_enc_attns[0][0][h:h + 1].squeeze(0).narrow(1, 0, len(mol_fragment)).narrow(0, 0, len(mol)),
        #         x=mol_fragment, y=mol,
        #         ax=axs[h])
        self.plot_heat_map(sum_dec_enc_attn.squeeze(0).narrow(1, 0, len(mol_fragment)).narrow(0, 0, len(mol)),
                           x=mol_fragment, y=mol, ax=axs)
        plt.savefig('./results/attn_heat_map.jpg')
        print('Plot complete!')


def train(train_loader, val_loader, optimizer, model, device='cpu', reverse_tgt_vocab={}):
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss1, loss2 = [], []
    qed_loss, valid_loss = 0, 0
    tqdm_bar = tqdm(train_loader, desc="Training")
    for enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs in tqdm_bar:
        enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs = enc_mol_inputs.to(device), enc_pro_inputs.to(
            device), dec_inputs.to(device), dec_outputs.to(device)
        outputs_frag, outputs_correct, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_mol_inputs, dec_inputs)
        valid_loss1 = criterion(outputs_frag, dec_outputs.view(-1))
        valid_loss2 = criterion(outputs_correct, dec_outputs.view(-1))
        loss1.append(valid_loss1.detach().cpu().numpy())
        loss2.append(valid_loss2.detach().cpu().numpy())
        optimizer.zero_grad()
        final_loss = valid_loss1 + valid_loss2
        final_loss.backward()
        optimizer.step()
        tqdm_bar.set_description(f"Loss: {valid_loss1:.4f} {valid_loss2:.4f}")
    train_avg_loss1 = (sum(loss1) / len(loss1))
    train_avg_loss2 = (sum(loss2) / len(loss2))
    train_avg_loss = (train_avg_loss1 + train_avg_loss2 / (train_avg_loss1 / train_avg_loss2))
    validate_loss = validate(val_loader, model, reverse_tgt_vocab, device=device)
    return model, optimizer, train_avg_loss, validate_loss, enc_self_attns, dec_self_attns, dec_enc_attns


def validate(loader, model, reverse_tgt_vocab, device='cpu'):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = []
    with torch.no_grad():
        tqdm_bar = tqdm(loader, desc="Validating")
        for enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs in tqdm_bar:
            enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs = enc_mol_inputs.to(device), enc_pro_inputs.to(
                device), dec_inputs.to(device), dec_outputs.to(device)
            outputs_frag, outputs_correct, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_mol_inputs, dec_inputs)
            valid_loss1 = criterion(outputs_frag, dec_outputs.view(-1))
            valid_loss2 = criterion(outputs_correct, dec_outputs.view(-1))
            loss.append((valid_loss1 + valid_loss2 / (valid_loss1 / valid_loss2)).detach().cpu().numpy())
            tqdm_bar.set_description(f"Loss: {valid_loss1:.4f} {valid_loss2:.4f}")
        return sum(loss) / len(loss)


def compute_tensor_qed_loss(smiles_tensor, smiles_count, reverse_tgt_vocab):
    true_qed = torch.empty(1, smiles_count)
    for index, i in enumerate(torch.chunk(smiles_tensor, smiles_count, dim=0)):
        smile = re.sub('EOS.*', '', flit_back(''.join([reverse_tgt_vocab[t.item()] for t in i.squeeze()])))
        qed = 0
        try:
            qed = QED.default(Chem.MolFromSmiles(smile))
        except Exception as e:
            pass
        true_qed[0, index] = qed
    true_qed = Variable(true_qed, requires_grad=True)
    need_qed = torch.ones(1, smiles_count)
    MSE = nn.MSELoss(reduction='mean')
    qed_loss = MSE(true_qed, need_qed)
    return qed_loss


def translate_one(model, src_vocab, pro_vocab, tgt_vocab,
                  mol_fragment='*C(=O)c1nc(*)c(-c2ccc(Cl)cc2)n1C',
                  mol='Cn1c(nc(c1-c1ccc(Cl)cc1)-c1ccc(Cl)cc1Cl)C(=O)NC1CCCCC1',
                  pro='MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS'):
    sentences = [[split_sentence(mol_fragment, 0), split_sentence(pro, 3), split_sentence(mol, 1),
                  split_sentence(mol, 2)]]
    enc_mol_inputs, enc_pro_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, pro_vocab, tgt_vocab)
    if constant.WITH_PROTEIN:
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_mol_inputs.to(device),
                                                                       enc_pro_inputs.to(device), dec_inputs.to(device))
    else:
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_mol_inputs.to(device), dec_inputs.to(device))
    return outputs, enc_self_attns, dec_self_attns, dec_enc_attns, mol_fragment, mol


def evaluate(pairs, model, src_vocab, pro_vocab, tgt_vocab, reverse_src_vocab, reverse_tgt_vocab, train_pairs_seed=1,
             flag=1, output_path='TransformerResults1123.txt', test_size=constant.TEST_SIZE):
    start = time.time()
    sentences = []
    if flag == 1:
        _, pairs = train_test_split(pairs, test_size=test_size, random_state=train_pairs_seed)
    else:
        pairs, _ = train_test_split(pairs, test_size=test_size, random_state=train_pairs_seed)
    for pair in pairs:
        sentences.append([split_sentence(pair[0], 0), split_sentence(pair[2], 3), split_sentence(pair[1], 1),
                          split_sentence(pair[1], 2)])
    print(sentences[0])
    loader = Data.DataLoader(MyDataSetLazy(sentences, src_vocab, pro_vocab, tgt_vocab), constant.BATCH_SIZE, True,
                             num_workers=6)

    for enc_mol_inputs, enc_pro_inputs, _, tgt in loader:
        if constant.WITH_PROTEIN:
            enc_mol_inputs = enc_mol_inputs.to(device)
            enc_pro_inputs = enc_pro_inputs.to(device)
            enc_mol_outputs, _ = model.mol_encoder(enc_mol_inputs)
            enc_pro_outputs, _ = model.pro_encoder(enc_pro_inputs)
        else:
            enc_mol_inputs = enc_mol_inputs.to(device)
            enc_mol_outputs, _ = model.encoder(enc_mol_inputs)

        dec_inputs = torch.full((enc_mol_inputs.size(0), 1), constant.SOS_TOKEN, dtype=enc_mol_inputs.dtype).to(device)
        terminal = torch.zeros(enc_mol_inputs.size(0), dtype=torch.bool).to(device)

        for _ in range(constant.MAX_LENGTH):
            if constant.WITH_PROTEIN:
                dec_outputs, _, _ = model.decoder(dec_inputs, enc_mol_inputs, enc_pro_inputs,
                                                  torch.cat([enc_mol_outputs, enc_pro_outputs], dim=1))
            else:
                dec_outputs, _, _ = model.decoder(dec_inputs, enc_mol_inputs, enc_mol_outputs)

            projected = model.projection(dec_outputs)
            next_symbols = projected[:, -1, :].max(dim=-1)[1]
            dec_inputs = torch.cat([dec_inputs, next_symbols.unsqueeze(1)], dim=1)

           
            terminal |= next_symbols == constant.EOS_TOKEN
            if terminal.all():
                break

        
        with open('{}{}'.format(constant.RESULT_PATH, output_path), 'a+') as f:
            for i in range(enc_mol_inputs.size(0)):
                res = flit_back(
                    re.sub('EOS.*', '', (''.join([reverse_tgt_vocab[t.item()] for t in dec_inputs[i, 1:]]))))
                src_part = flit_back(
                    (''.join([reverse_src_vocab[t.item()] for t in enc_mol_inputs[i]])).replace('PAD', ''))
                src_whole = flit_back(
                    re.sub('EOS.*', '', (''.join([reverse_tgt_vocab[t.item()] for t in tgt[i]]))))
                f.writelines(src_whole + '\t' + src_part + '\t' + res + '\n')

    end = time.time()
    print("evaluate complete, cost time {} s".format(end - start))
