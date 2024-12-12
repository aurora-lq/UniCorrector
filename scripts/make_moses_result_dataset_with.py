from rdkit import Chem


def deal(filepath, filename):
    with open(filepath + filename, 'r') as f:
        with open("temp_" + filename, 'w') as ff:
            f.readline()
            lines = f.readlines()
            for line in lines:
                smiles = line.strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(smiles,flush=True)


#deal("", "vae_generated_guaca.csv")
deal("", "2_1_false.txt")
