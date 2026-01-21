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

## Main features

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
```

## ⚙️ Installation
pip install -r requirements.txt
PyTorch should be installed separately according to your CUDA environment.
