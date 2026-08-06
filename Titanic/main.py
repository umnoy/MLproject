import os
import numpy as np
import pandas as pd
import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import]
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from omegaconf import OmegaConf  # type: ignore[import]
from data import prepare_titanic_data


# ==========================================
# UTILS & SEED
# ==========================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==========================================
# DATASET
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float, output_dim: int = 1):
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


# ==========================================
# TRAIN & EVAL LOOPS
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(X_batch)
    return running_loss / len(dataloader.dataset)


def eval_model(model, dataloader, device):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_list.append(preds.cpu().numpy())
    return np.vstack(preds_list)

# ==========================================
# MAIN PIPELINE
# ==========================================
def main(config_path: str = "config.yaml"):
    # 1. Загрузка конфигурации
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден по пути {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.paths.output_dir, "config.yaml"))
    
    seed_everything(cfg.general.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"--> Запуск эксперимента: {cfg.general.experiment_name} на {device}")

    # 2. Загрузка данных
    if os.path.exists(cfg.paths.train_csv):
        X_df, y_df, X_test_df = prepare_titanic_data(cfg.paths.train_csv, cfg.paths.test_csv)
    else: 
        raise FileNotFoundError(f"Не найден {cfg.paths.train_csv}")

    # Переводим в NumPy массивы, чтобы KFold и скейлер работали без проблем
    X = X_df.values
    y = y_df.values
    X_test = X_test_df.values

    # 3. Настройка кросс-валидации
    task_type = cfg.general.task_type
    n_splits = cfg.split.n_splits
    oof_predictions = np.zeros_like(y, dtype=float)

    if task_type == "binary":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.general.seed)
        criterion = nn.BCEWithLogitsLoss()
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.general.seed)
        criterion = nn.MSELoss()

    # 4. Цикл обучения по фолдам
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y if task_type == "binary" else None)):
        print(f"\n[Fold {fold + 1}/{n_splits}]")
        
        # Нарезаем исходные X и y на текущий фолд
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        # Масштабирование: фитим скейлер ТОЛЬКО на трейне фолда (X_tr)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        
        # Передаем в датасеты ИМЕННО нарезанные X_tr и X_va
        train_ds = TabularDataset(X_tr, y_tr)
        val_ds = TabularDataset(X_va, y_va)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.training.batch_size, 
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.training.batch_size, 
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory
        )
        
        # Инициализация модели (передаем ширину X.shape[1])
        model = SimpleMLP(
            input_dim=X.shape[1],
            hidden_dims=cfg.model.params.hidden_dims,
            dropout_rate=cfg.model.params.dropout_rate
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.training.lr, 
            weight_decay=cfg.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.training.epochs
        )
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(cfg.paths.output_dir, f"model_fold_{fold}.pt")

        for epoch in range(cfg.training.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            
            # Валидация
            val_preds = eval_model(model, val_loader, device)
            
            if task_type == "binary":
                val_loss = criterion(
                    torch.tensor(val_preds), 
                    torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)
                ).item()
            else:
                val_loss = mean_squared_error(y_va, val_preds)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        # OOF предсказания для текущего фолда
        model.load_state_dict(torch.load(best_model_path))
        fold_oof = eval_model(model, val_loader, device)
        
        if task_type == "binary":
            # Применяем сигмоиду, так как логиты выходят из сетки без активации
            fold_oof = 1 / (1 + np.exp(-fold_oof))
            
        oof_predictions[val_idx] = fold_oof.squeeze()
        print(f"Fold {fold + 1} Best Loss: {best_val_loss:.5f}")

    # 5. Итоговая метрика OOF по всему датасету
    print("\n================ FINAL OOF EVALUATION ================")
    if task_type == "binary":
        score = roc_auc_score(y, oof_predictions)
        print(f"OOF ROC-AUC: {score:.5f}")
        score2 = accuracy_score(y, (oof_predictions > 0.5).astype(int))
        print(f"OOF Accuracy: {score2:.5f}")
    else:
        score = np.sqrt(mean_squared_error(y, oof_predictions))
        print(f"OOF RMSE: {score:.5f}")


if __name__ == "__main__":
    main("config.yaml")