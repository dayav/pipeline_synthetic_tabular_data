import numpy as np, pandas as pd, torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import trange

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS      = 1500          # self‑supervised epochs
BATCH_SIZE  = 2048
EMB_DIM     = 32          # TabNet n_d = n_a
SEED        = 42

# ---------------------------------------------------------------------
def make_tabnet_embedder(df: pd.DataFrame,
                         num_cols: list[str],
                         cat_cols: list[str]):
    """
    Returns  embed_ft(df_batch) -> np.ndarray[batch, EMB_DIM].

    • Un‑labelled pre‑training (works even with missing `y`)
    • Scales to > 100 k rows (mini‑batch)
    • Requires pytorch‑tabnet ≥ 4.0
    """

    # -------- 1. preprocessing --------------------------------------
    scaler = MinMaxScaler().fit(df[num_cols])
    enc    = OrdinalEncoder(handle_unknown='use_encoded_value',
                            unknown_value=-1).fit(df[cat_cols])

    def encode(df_):
        Xnum = scaler.transform(df_[num_cols]).astype(np.float32)
        Xcat = enc.transform(df_[cat_cols]).astype(np.int64) + 1  # 0 pad
        return np.hstack([Xnum, Xcat])

    X = encode(df)

    n_num = len(num_cols)
    cat_idxs = list(range(n_num, n_num + len(cat_cols)))
    cat_dims = [len(enc.categories_[i]) + 1 for i in range(len(cat_cols))]

    # -------- 2. self‑supervised pre‑training -----------------------
    pretrainer = TabNetPretrainer(
        n_d=EMB_DIM, n_a=EMB_DIM,
        n_steps=5,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=8,
        optimizer_params=dict(lr=1e-3),
        mask_type='sparsemax',          # same as paper
        device_name=DEVICE,
        seed=SEED,
        verbose=0)

    pretrainer.fit(
        X_train=X,
        eval_set=[X],
        max_epochs=EPOCHS,
        patience=15,
        batch_size=BATCH_SIZE,
        virtual_batch_size=256,
        drop_last=False)

    pretrainer.network.eval()

    # -------- 3. embedding closure ----------------------------------
    @torch.no_grad()
    def embed_ft(batch_df: pd.DataFrame) -> np.ndarray:
        Xb = torch.from_numpy(encode(batch_df)).to(DEVICE)
        # Forward through encoder only: returns (out, embedded_x, obf_vars)
        _, embedded_x, _ = pretrainer.network(Xb)
        # embedded_x shape : (batch, EMB_DIM)
        return embedded_x.cpu().numpy().astype(np.float32)

    return embed_ft
