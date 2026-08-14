import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from omegaconf import OmegaConf
from data import prepare_data
from model import MLP, TabularDataset
import optuna 
from omegaconf import OmegaConf

DATASETS = {
    'house_prices': prepare_data,
    # 'titanic': prepare_titanic_data,  -- вернуть, когда появится реальный импорт функции
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # просто в привычку на случай нескольких gpu


def train(cfg, device, X, y):
    # обучает по 1 модели на фолд, сохраняет лучший чекпоинт каждого фолда,
    # возвращает OOF-предсказания для честной оценки качества (RMSE на логарифме таргета)
    oof_predictions = np.zeros(len(y))
    fold_rmses = []

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

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if cfg.general.task_type == "binary" else nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.training.scheduler_epochs, cfg.training.scheduler_rate)

        best_rmse = float('inf')
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
            scheduler.step()

            model.eval()
            val_preds_list = []
            val_targets_list = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    preds = model(X_batch)
                    val_preds_list.append(preds.cpu().numpy())
                    val_targets_list.append(y_batch.numpy())

            predi = np.concatenate(val_preds_list).squeeze()
            targ = np.concatenate(val_targets_list).squeeze()

            rmse = np.sqrt(mean_squared_error(targ, predi))
            
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), best_model_path)

            print(f'fold {fold}, epoch {epoch}, rmse {rmse:.5f}')

        fold_rmses.append(best_rmse)

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        fold_preds_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                fold_preds_list.append(preds.cpu().numpy())

        fold_preds = np.concatenate(fold_preds_list).squeeze()
        oof_predictions[val_idx] = fold_preds

    final_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    print()
    print(f"OOF rmse: {final_rmse:.5f}")
    print(f"CV rmse: {np.mean(fold_rmses):.5f}, std: {np.std(fold_rmses).round(5)}")

    metrics_path = os.path.join(cfg.paths.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"CV rmse: {np.mean(fold_rmses):.4f} std:{np.std(fold_rmses):.4f}\n")
        f.write(f"OOF rmse: {final_rmse:.4f}\n")

    np.save(os.path.join(cfg.paths.output_dir, "mlp_oof.npy"), oof_predictions)

    return oof_predictions


def predict(cfg, device, X, X_test_df, id_column_values):
    scaler_full = StandardScaler()
    scaler_full.fit_transform(X)
    X_test_scaled = scaler_full.transform(X_test_df.values)

    test_ds = TabularDataset(X_test_scaled)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    test_preds_sum = np.zeros(len(X_test_df))

    for fold in range(cfg.split.n_splits):  # усреднение пяти фолдовых моделей
        model = MLP(input_dim=X.shape[1], hidden_dims=cfg.model.params.hidden_dims, dropout_rate=cfg.model.params.dropout_rate).to(device)
        model_path = os.path.join(cfg.paths.output_dir, f"model_fold_{fold}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_test_preds = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                fold_test_preds.append(preds.cpu().numpy())

        fold_test_preds = np.concatenate(fold_test_preds).squeeze()
        test_preds_sum += fold_test_preds

    test_preds_avg = test_preds_sum / cfg.split.n_splits

    np.save(os.path.join(cfg.paths.output_dir, "mlp_test_preds.npy"), test_preds_avg)

    submission = pd.DataFrame({
        cfg.general.id_column: id_column_values,
        "SalePrice": np.exp(test_preds_avg), 
    })
    submission.to_csv(os.path.join(cfg.paths.output_dir, "submission.csv"), index=False)

    return test_preds_avg


def main(config_path="config.yaml"):
    cfg = OmegaConf.load(config_path)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.paths.output_dir, "config.yaml"))

    seed_everything(cfg.general.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"{cfg.general.experiment_name}, {device}")

    prepare_func = DATASETS.get(cfg.general.dataset)
    if prepare_func is None:
        raise ValueError(f"поправь название датасета в конфиге: {cfg.general.dataset}")

    X_df, y_df, X_test_df, _, id_column_values = prepare_func(
        cfg.paths.train_csv, cfg.paths.test_csv, cfg.general.id_column, dummies=True, scale=False
    )
    X = X_df.values
    y = y_df.values.astype(np.float32)# MSELoss ждёт ту же форму, что и выход модели

    train(cfg, device, X, y)
    predict(cfg, device, X, X_test_df, id_column_values)


if __name__ == "__main__":
    main()