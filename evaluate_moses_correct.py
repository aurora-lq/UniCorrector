import sys

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
import utils.sascore as sascore
from rdkit.Chem.Crippen import MolLogP
import matplotlib.pyplot as plt
import seaborn as sns
import time

file_dir = ""
source_file = "./datasets/dataset_v1.csv"
file = file_dir+"inference.txt"
output_file = open(file_dir+'evaluate_inference.txt', 'w')


def evaluate_result(data_file):
    qed_differ = []
    with open(data_file, 'r') as f:
        all_mol = set()
        novelty_mol = set()
        with open(source_file, 'r') as ff:
            ff.readline()
            lines = ff.readlines()
            for line in lines:
                all_mol.add(line.split(',')[0])
        print(len(all_mol))
        lines = f.readlines()
        generated_qed_list, generated_log_p_list, generated_sa_score_list = [], [], []
        source_qed_list, source_log_p_list, source_sa_score_list = [], [], []
        source_generated_dict = {}
        generated_mol_smiles_set, generated_mol_smiles_list = set(), []
        similarity = []
        recurrence = 0
        for line in lines:
            pair = line.split()
            if len(pair) != 3:
                print(pair)
                continue
            if pair[0] == pair[2]:
                recurrence += 1
            #if source_mol := Chem.MolFromSmiles(pair[0]):
            source_mol = Chem.MolFromSmiles(pair[0])
            if source_mol:
                try:
                    source_qed = QED.default(source_mol)
                    source_qed_list.append(source_qed)
                except Exception as e:
                    continue
                source_log_p_list.append(MolLogP(source_mol))
                source_sa_score_list.append(sascore.calculateScore(source_mol))
                #if generated_mol := Chem.MolFromSmiles(pair[2]):
                generated_mol = Chem.MolFromSmiles(pair[2])
                if generated_mol:
                    try:
                        generated_qed = QED.default(generated_mol)
                        generated_qed_list.append(generated_qed)
                    except Exception as e:
                        continue
                    generated_mol_smiles_set.add(pair[2])
                    generated_mol_smiles_list.append(pair[2])
                    if source_generated_dict.get((pair[0], source_qed)):
                        source_generated_dict[(pair[0], source_qed)].append((pair[2], generated_qed))
                    else:
                        source_generated_dict[(pair[0], source_qed)] = [(pair[2], generated_qed)]
                    sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(source_mol),
                                                            Chem.RDKFingerprint(generated_mol))
                    if (generated_qed - source_qed) > 0.4 and sim > 0.8:
                        qed_differ.append((pair[0], source_qed, pair[2], generated_qed))
                    generated_log_p_list.append(MolLogP(generated_mol))
                    generated_sa_score_list.append(sascore.calculateScore(generated_mol))
                    similarity.append(sim)
                    if pair[2] not in all_mol:
                        novelty_mol.add(pair[2])
        # assert len(lines) == len(source_qed_list)
        print("Source mol number is: ", len(source_qed_list))
        print("All generated SMILES number is: ", len(lines))
        #print("Source mol average qed and std_qed: ", np.mean(source_qed_list), np.std(source_qed_list))
        #print("Source mol average LogP: ", np.mean(source_log_p_list), np.std(source_log_p_list))
        #print("Source mol average SaScore: ", np.mean(source_sa_score_list), np.std(source_sa_score_list))
        print("Generated mol average qed: ", np.mean(generated_qed_list), np.std(generated_qed_list))
        print("Generated mol average LogP: ", np.mean(generated_log_p_list), np.std(generated_log_p_list))
        print("Generated mol average SaScore: ", np.mean(generated_sa_score_list), np.std(generated_sa_score_list))
        print("Generated mol number: ", len(generated_log_p_list))
        print("Valid Generated mol percent1: ", len(generated_log_p_list) / len(lines))
        print("Valid Generated mol percent: ", len(generated_log_p_list) / len(source_log_p_list))
        print("Uniquness@1k: ",len(set(generated_mol_smiles_list[:1000]))/len(generated_mol_smiles_list[:1000]))
        print("Novelty mol number: ", len(novelty_mol))
        print("Uniquness mol number: ", len(generated_mol_smiles_set))
        print("Novelty mol percent: ", len(novelty_mol)/len(generated_mol_smiles_set))
        print("Source and Generated mol similarity: ", np.mean(similarity), np.std(similarity))
        print("Generated Recurrence rate is :", recurrence / len(source_qed_list))
        for i in qed_differ:
            print(str(i))
        for k, v in source_generated_dict.items():
            with open(file_dir+'source_generated_dict.txt', 'a+') as ff:
                ff.write(str(k) + '\t' + str(v) + '\n')

        ax1 = sns.kdeplot(source_qed_list, color='blue', shade=True, label="source_qed")
        ax1.legend(loc="upper right")
        ax2 = sns.kdeplot(generated_qed_list, color='orange', shade=True, label="generated_qed")
        ax2.legend(loc="upper right")
        plt.suptitle('QED')
        plt.savefig(file_dir+'QED.jpg')
        plt.show()
        ax3 = sns.kdeplot(source_log_p_list, color='blue', shade=True, label="source_log_p")
        ax3.legend(loc="upper right")
        ax4 = sns.kdeplot(generated_log_p_list, color='orange', shade=True, label="generated_log_p")
        ax4.legend(loc="upper right")
        plt.suptitle('LogP')
        plt.savefig(file_dir+'LogP.jpg')
        plt.show()
        ax5 = sns.kdeplot(source_sa_score_list, color='blue', shade=True, label="source_sa_score")
        ax5.legend(loc="upper right")
        ax6 = sns.kdeplot(generated_sa_score_list, color='orange', shade=True, label="generated_SAscore")
        ax6.legend(loc="upper right")
        plt.suptitle('SAscore')
        plt.savefig(file_dir+'SAscore.jpg')
        plt.show()


start = time.time()
sys.stdout = output_file
evaluate_result(file)
end = time.time()
print(end - start, 's')
output_file.close()
