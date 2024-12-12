from rdkit import Chem


def deal(filepath, filename):
    with open(filepath + filename, 'r') as f:
        with open("/home/smh/D3hit2lead/datasets/dataset_" + filename, 'w') as ff:
            f.readline()
            lines = f.readlines()
            for line in lines:
                smiles = line.strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    ff.write('C\t' + smiles + '\t' + 'M\n')


deal("", "23_false.txt")
