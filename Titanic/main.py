import os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from omegaconf import OmegaConf  
from data import prepare_titanic_data


DATASETS = {
    'titanic': prepare_titanic_data,
}

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]  

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) #просто в привычку на случай нескольких gpu


def main(config_path="config.yaml"):
    cfg = OmegaConf.load(config_path)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    seed_everything(cfg.general.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"{cfg.general.experiment_name}, {device}")

    prepare_func = DATASETS.get(cfg.general.dataset)
    if prepare_func is None:
        raise ValueError(f"поправь название датасета в конфиге: {cfg.general.dataset}")

    X_df, y_df, X_test_df = prepare_func(cfg.paths.train_csv, cfg.paths.test_csv)

    X = X_df.values
    y = y_df.values

    oof_predictions = np.zeros(len(y))

    if cfg.general.task_type == "binary":
        kf = StratifiedKFold(n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.general.seed)
        split_target = y
    else:
        kf = KFold(n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.general.seed)
        split_target = None


    for fold, (train_idx, val_idx) in enumerate(kf.split(X, split_target)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr) 
        X_va = scaler.transform(X_va) 

        train_ds = TabularDataset(X_tr, y_tr)
        val_ds = TabularDataset(X_va, y_va)

        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

        model = MLP(input_dim=X_tr.shape[1], hidden_dims=cfg.model.params.hidden_dims, dropout_rate=cfg.model.params.dropout_rate).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = cfg.training.lr, weight_decay= cfg.training.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if cfg.general.task_type == "binary" else nn.MSELoss()

        best_accuracy = 0
        best_model_path = os.path.join(cfg.paths.output_dir, f"model_fold_{fold}.pt")

        for epoch in range(cfg.training.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds_list = []
            val_targets_list = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    preds = model(X_batch)
                    val_preds_list.append(preds.cpu().numpy())
                    val_targets_list.append(y_batch.numpy())

            predi = np.concatenate(val_preds_list)
            targ = np.concatenate(val_targets_list)

            probs = 1 / (1 + np.exp(-predi))# sigmoid вручную
            preds_class = (probs > 0.5).astype(int)
            accuracy = accuracy_score(targ, preds_class)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)

            print(f'fold {fold}, epoch {epoch}, accuracy {accuracy}')


        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        fold_preds_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                fold_preds_list.append(preds.cpu().numpy())

        fold_preds = np.concatenate(fold_preds_list)
        fold_probs = 1 / (1 + np.exp(-fold_preds))

        oof_predictions[val_idx] = fold_probs.squeeze()
    final_accuracy = accuracy_score(y, (oof_predictions > 0.5).astype(int))
    print(f"OOF accuracy: {final_accuracy}")




if __name__ == "__main__":
    main()