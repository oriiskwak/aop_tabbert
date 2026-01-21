<h1 align="center">aop_tabbert</h1>
<p align="center">
  <b>OECD TG hitcall (0/1) prediction using TabularBERT</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA%20supported-red" />
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green" />
  <img src="https://img.shields.io/badge/Domain-OECD%20TG-orange" />
</p>

---

This repository provides a **TabularBERT-based training pipeline** for predicting  
**hitcall (binary: 0 / 1)** using OECD Test Guideline (TG) experimental datasets.

### Main features

- Combination of descriptor features and SMILES PCA features  
- RandomForest-based feature selection  
- TabularBERT pretraining → finetuning pipeline  
- Class imbalance handling with Focal Loss, class weights, and F1 threshold tuning  

---

## 📁 Repository structure

```text
aop_tabbert/
├─ datasets/        # Input datasets (not included in repo)
├─ pretraining/    # TabularBERT pretraining related modules
├─ fine-tuning/    # Finetuning utilities
├─ tabularbert/    # TabularBERT source code (local copy / modified)
├─ opt_hitcall.py  # Main experiment script (entry point)
├─ requirements    # Python dependencies
└─ setup.py        # Package setup
⚙️ Installation
bash
코드 복사
pip install -r requirements.txt
PyTorch should be installed separately according to your CUDA environment.

📥 Input dataset
The input dataset should be placed as follows:

text
코드 복사
datasets/
 └─ OECD TG XXX_embedded_num.csv
Required columns
Column name	Description
No	Sample ID
OECD TG XXX	Target label (0 / 1 / NaN)
PC_*	SMILES PCA features
Others	Descriptor features

Samples with missing targets (NaN) are automatically excluded from training.

🛠 Configure target TG
Edit the top part of opt_hitcall.py:

python
코드 복사
DATA_PATH = "./datasets/OECD TG 487_embedded_num.csv"
TARGET_COL = "OECD TG 487"
Select one of the recommended hyperparameter blocks depending on the TG.

python
코드 복사
# TG 487 / 471
TOP_K_LIST = [60, 65, 70, 120, 130, 140, 150]
PC_DIM_LIST = [10, 12, 13, 15, 20]
▶️ Run experiment
bash
코드 복사
python opt_hitcall.py
During execution:

All (desc_topk, pc_dim) combinations are automatically grid-searched

Each configuration is trained and evaluated

At the end, the best configuration and classification report are printed

📊 Output
For each configuration, the following metrics are reported:

test_F1

AUC

Precision, Recall

val_th (best threshold from validation)

w_pos (positive class weight)

Finally, a best configuration summary based on test F1-score is printed.
