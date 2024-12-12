UniCorrector: A Universal Model Corrects Invalid SMILES and Enhances Corrected Molecular Properties

1 Requirements
python 3.7.15
torch 1.12.0
rdkit 2023.3.2
pandas 1.3.5
numpy 1.21.5


2 Data
See dataset fold to get training data. 
Data format: The first column contains valid SMILES, the second column contains invalid SMILES, the third column is a padding character, and the fourth to sixth columns represent the properties qed, logp, and sascore.

3 Training
python data_preprocess.py. More arguments setting can be found and changed in code file data_preprocess.py. Modify the dataset dataset_path = '', which can be either a training dataset or an inference dataset.

4 Inferencing
python data_preprocess.py. You need to modify the parameter constant.INFERENCE = True in ./config/config.py.

5 Evaluating
python evaluate_moses_correct.py. File_dir and source_file: The first specifies the path to the folder to be evaluated, and the second refers to the source file for comparison.

6 Folder Description
results：
modelcuda/0.pth refers to the model trained using other dataset construction methods.
modelcuda/3.pth refers to the model trained without using error types.
epoch100modelcuda/3.pth refers to the model trained using error types.
logpepoch100modelcuda/0.pth refers to the model trained with conditional tokens for the logp property.
promodelcuda/1.pth refers to the model trained with conditional tokens for the qed property.
scripts:
Refers to the script files used for constructing the dataset.
