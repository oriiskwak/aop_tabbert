import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from tabularbert import TabularBERTTrainer
from tabularbert.utils.metrics import ClassificationError

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)



# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss



# Utils
def threshold_grid_search(y_true, proba, th_min=0.05, th_max=0.80, n=151):
    ths = np.linspace(th_min, th_max, n)
    best_f1, best_th = -1.0, 0.5
    for th in ths:
        pred = (proba >= th).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
    return best_th, best_f1


def quantile_clip_fit(train_df, q_low=0.001, q_high=0.999):
    lo = train_df.quantile(q_low, numeric_only=True)
    hi = train_df.quantile(q_high, numeric_only=True)
    return lo, hi


def quantile_clip_apply(df, lo, hi):
    return df.clip(lower=lo, upper=hi, axis=1)


def fit_preprocessor(
    desc_tr_df, pc_tr_df, y_tr,
    desc_all_df, pc_all_df,
    top_k_desc, pc_dim,
    lo, hi,
    w_cap=2.0,
):
    # DESC clip
    desc_tr_c = quantile_clip_apply(desc_tr_df.copy(), lo, hi)
    desc_all_c = quantile_clip_apply(desc_all_df.copy(), lo, hi)

    # DESC impute (fit on train)
    imp_desc = SimpleImputer(strategy="median")
    desc_tr_imp = imp_desc.fit_transform(desc_tr_c).astype(np.float64)
    desc_all_imp = imp_desc.transform(desc_all_c).astype(np.float64)

    # DESC feature selection (fit on train)
    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(desc_tr_imp, y_tr)

    k_eff = min(top_k_desc, desc_tr_imp.shape[1])
    idx_desc = np.argsort(rf.feature_importances_)[-k_eff:]

    # PC impute + PCA (fit on train)
    if pc_dim > 0 and pc_tr_df.shape[1] > 0:
        imp_pc = SimpleImputer(strategy="median")
        pc_tr_imp = imp_pc.fit_transform(pc_tr_df).astype(np.float64)
        pc_all_imp = imp_pc.transform(pc_all_df).astype(np.float64)

        pc_eff = min(pc_dim, pc_tr_imp.shape[1])
        pca = PCA(n_components=pc_eff, random_state=42)
        pc_tr_red = pca.fit_transform(pc_tr_imp)
        pc_all_red = pca.transform(pc_all_imp)
    else:
        imp_pc = None
        pca = None
        pc_eff = 0
        pc_tr_red = np.zeros((len(desc_tr_df), 0), dtype=np.float64)
        pc_all_red = np.zeros((len(desc_all_df), 0), dtype=np.float64)

    # build train/all BEFORE scaling
    desc_tr_sel = desc_tr_imp[:, idx_desc]
    desc_all_sel = desc_all_imp[:, idx_desc]

    X_tr = np.hstack([desc_tr_sel, pc_tr_red])
    X_all = np.hstack([desc_all_sel, pc_all_red])

    # scale (fit on train)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_all_sc = scaler.transform(X_all)

    # class weights (cap)
    num_neg = np.sum(y_tr == 0)
    num_pos = np.sum(y_tr == 1)
    w_pos = num_neg / (num_pos + 1e-8)
    w_pos = min(w_pos, w_cap)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float)

    preproc = {
        "lo": lo, "hi": hi,
        "imp_desc": imp_desc,
        "idx_desc": idx_desc,
        "imp_pc": imp_pc,
        "pca": pca,
        "scaler": scaler,
        "k_eff": int(k_eff),
        "pc_eff": int(pc_eff),
        "w_pos": float(w_pos),
    }
    return preproc, X_tr_sc, X_all_sc, class_weights


def transform_with_preprocessor(preproc, desc_df, pc_df):
    # clip desc
    desc_c = quantile_clip_apply(desc_df.copy(), preproc["lo"], preproc["hi"])

    # impute desc
    desc_imp = preproc["imp_desc"].transform(desc_c).astype(np.float64)
    desc_sel = desc_imp[:, preproc["idx_desc"]]

    # smiles pca
    if preproc["pc_eff"] > 0 and pc_df.shape[1] > 0:
        pc_imp = preproc["imp_pc"].transform(pc_df).astype(np.float64)
        pc_red = preproc["pca"].transform(pc_imp)
    else:
        pc_red = np.zeros((len(desc_df), 0), dtype=np.float64)

    X = np.hstack([desc_sel, pc_red])
    X_sc = preproc["scaler"].transform(X)
    return X_sc
def train_tabularbert(
    X_all_pre, X_tr, y_tr, X_va, y_va, device,
    num_bins=30, embedding_dim=64, n_layers=2, n_heads=4,
    pretrain_epochs=30, finetune_epochs=120, patience=20,
    class_weights=None, focal_gamma=2.0,
):
    trainer = TabularBERTTrainer(x=X_all_pre, num_bins=num_bins, device=device)
    trainer.set_bert(embedding_dim=embedding_dim, n_layers=n_layers, n_heads=n_heads)

    # Pretrain
    trainer.pretrain(epochs=pretrain_epochs, lamb=0.5)

    # Finetune
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=focal_gamma)

    trainer.finetune(
        x=X_tr, y=y_tr,
        valid_x=X_va, valid_y=y_va,
        epochs=finetune_epochs, batch_size=32,
        lamb=0.1, penalty="L2",
        criterion=criterion,
        metric=ClassificationError(),
        num_workers=0,
        patience=patience
    )
    return trainer


def predict_proba(trainer, X, device):
    trainer.model.eval()
    bins = trainer.discretizer.discretize(X)
    with torch.no_grad():
        logits = trainer.model(torch.tensor(bins, dtype=torch.long).to(device))
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return proba


def run_one_config(
    desc_tr_df, desc_va_df, desc_te_df, desc_all_df,
    pc_tr_df,   pc_va_df,   pc_te_df,   pc_all_df,
    y_tr, y_va, y_te,
    device,
    top_k_desc,
    pc_dim,
    num_bins=30,
    w_cap=2.0,
    th_min=0.05, th_max=0.80, th_n=151,
):
    lo, hi = quantile_clip_fit(desc_tr_df)

    preproc, X_tr, X_all_pre, class_weights = fit_preprocessor(
        desc_tr_df, pc_tr_df, y_tr,
        desc_all_df, pc_all_df,
        top_k_desc=top_k_desc,
        pc_dim=pc_dim,
        lo=lo, hi=hi,
        w_cap=w_cap,
    )

    X_va = transform_with_preprocessor(preproc, desc_va_df, pc_va_df)
    X_te = transform_with_preprocessor(preproc, desc_te_df, pc_te_df)

    trainer = train_tabularbert(
        X_all_pre=X_all_pre,
        X_tr=X_tr, y_tr=y_tr,
        X_va=X_va, y_va=y_va,
        device=device,
        num_bins=num_bins,
        embedding_dim=64, n_layers=2, n_heads=4,
        pretrain_epochs=30, finetune_epochs=120,
        patience=20,
        class_weights=class_weights,
        focal_gamma=2.0,
    )

    # threshold는 VAL에서만!!
    va_proba = predict_proba(trainer, X_va, device)
    val_th, val_f1 = threshold_grid_search(y_va, va_proba, th_min=th_min, th_max=th_max, n=th_n)

    te_proba = predict_proba(trainer, X_te, device)
    te_auc = float(roc_auc_score(y_te, te_proba))
    te_pred = (te_proba >= val_th).astype(int)
    te_f1 = float(f1_score(y_te, te_pred))

    tp = int(((te_pred == 1) & (y_te == 1)).sum())
    fp = int(((te_pred == 1) & (y_te == 0)).sum())
    fn = int(((te_pred == 0) & (y_te == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    rep = classification_report(y_te, te_pred, target_names=["Neg", "Pos"], zero_division=0)

    return {
        "desc_k": preproc["k_eff"],
        "pc_dim": preproc["pc_eff"],
        "w_pos": preproc["w_pos"],
        "val_th": float(val_th),
        "val_f1": float(val_f1),
        "test_f1": float(te_f1),
        "auc": float(te_auc),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "report": rep,
    }


#data
#OECD 마다 리스트 값을 다르게 줘야함 (최적의 값 재현)
def main():
    DATA_PATH = "./datasets/OECD TG 487_embedded_num.csv"
    TARGET_COL = "OECD TG 487"

# 487,471 시 주석해제
    TOP_K_LIST = [60, 65, 70, 120, 130, 140, 150]   
    PC_DIM_LIST = [10, 12, 13, 15, 20]  


# 476 시 주석해제
    # TOP_K_LIST = [20, 30, 50, 80, 120, 160]   
    # PC_DIM_LIST = [0, 2, 5, 10]   

#475,421 시 주석해제
    # TOP_K_LIST = [30, 40, 50, 60]   
    # PC_DIM_LIST = [0, 2, 5] 

    NUM_BINS = 30                      

    W_CAP = 2.0
    TH_MIN, TH_MAX, TH_N = 0.05, 0.80, 151

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(DATA_PATH)

    y_all = data[TARGET_COL].fillna(-1).values.astype(int)

    full_feat = data.drop(columns=["No", TARGET_COL]).copy()
    pc_all = full_feat.loc[:, full_feat.columns.str.startswith("PC_")].copy()
    desc_all = full_feat.loc[:, ~full_feat.columns.str.startswith("PC_")].copy()

    desc_all = desc_all.replace([np.inf, -np.inf], np.nan)
    pc_all = pc_all.replace([np.inf, -np.inf], np.nan)
    if "Ipc" in desc_all.columns:
        desc_all["Ipc"] = np.log1p(desc_all["Ipc"].clip(lower=0))

    labeled_mask = (y_all != -1)
    y_lab = y_all[labeled_mask]
    desc_lab = desc_all.loc[labeled_mask].copy()
    pc_lab = pc_all.loc[labeled_mask].copy()

    # train/val/test split 
    desc_tr, desc_te, pc_tr, pc_te, y_tr, y_te = train_test_split(
        desc_lab, pc_lab, y_lab, test_size=0.2, random_state=42, stratify=y_lab
    )
    desc_tr, desc_va, pc_tr, pc_va, y_tr, y_va = train_test_split(
        desc_tr, pc_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    results = []

    for topk in TOP_K_LIST:
        for pcd in PC_DIM_LIST:
            print("\n" + "=" * 78)
            print(f"[RUN] desc_topk={topk} | pc_dim={pcd} | num_bins={NUM_BINS}")

            out = run_one_config(
                desc_tr, desc_va, desc_te, desc_all,
                pc_tr,   pc_va,   pc_te,   pc_all,
                y_tr, y_va, y_te,
                device=device,
                top_k_desc=topk,
                pc_dim=pcd,
                num_bins=NUM_BINS,
                w_cap=W_CAP,
                th_min=TH_MIN, th_max=TH_MAX, th_n=TH_N,
            )
            results.append(out)

            print(
                f"desc_k={out['desc_k']:>3} | pc_dim={out['pc_dim']:>2} | "
                f"test_F1={out['test_f1']:.4f} | AUC={out['auc']:.4f} | "
                f"P={out['precision']:.4f} | R={out['recall']:.4f} | "
                f"val_th={out['val_th']:.4f} | w_pos={out['w_pos']:.2f}"
            )

    print("\n" + "#" * 78)
    print("SUMMARY (higher test_F1 is better)")
    results_sorted = sorted(results, key=lambda d: d["test_f1"], reverse=True)
    for r in results_sorted[:20]:
        print(
            f"pc_dim={r['pc_dim']:>2} | desc_k={r['desc_k']:>3} | "
            f"test_F1={r['test_f1']:.4f} | AUC={r['auc']:.4f} | "
            f"P={r['precision']:.4f} | R={r['recall']:.4f} | "
            f"val_th={r['val_th']:.4f} | w_pos={r['w_pos']:.2f}"
        )

    best = results_sorted[0]
    print("\n" + "=" * 78)
    print(
        f"BEST CONFIG: pc_dim={best['pc_dim']} | desc_k={best['desc_k']} | "
        f"test_F1={best['test_f1']:.4f} | AUC={best['auc']:.4f} | val_th={best['val_th']:.4f}"
    )
    print(best["report"])


if __name__ == "__main__":
    main()
