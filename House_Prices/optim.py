import os
import numpy as np
import torch
import optuna
from omegaconf import OmegaConf

from main import seed_everything, train, predict, DATASETS


def objective(trial, base_cfg, device, X, y, search_output_dir):
    cfg = OmegaConf.merge(base_cfg, {})  # копия, не мутируем base_cfg

    cfg.model.params.hidden_dims = trial.suggest_categorical(
        "hidden_dims", [[64, 32], [32, 16], [128, 64, 32], [64]]
    )
    cfg.model.params.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    cfg.training.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    cfg.training.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # урезаем для скорости поиска — на финальное дообучение это не влияет,
    # там берём полные значения из base_cfg
    cfg.split.n_splits = 3
    cfg.training.epochs = min(cfg.training.epochs, 15)

    # свой output_dir на каждый trial, чтобы чекпоинты не затирали друг друга
    cfg.paths.output_dir = os.path.join(search_output_dir, f"trial_{trial.number}")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    oof_predictions = train(cfg, device, X, y)
    rmse = np.sqrt(np.mean((y - oof_predictions) ** 2))
    return rmse


def main(config_path="config.yaml", n_trials=30):
    base_cfg = OmegaConf.load(config_path)
    seed_everything(base_cfg.general.seed)
    device = torch.device(base_cfg.training.device if torch.cuda.is_available() else "cpu")

    search_output_dir = os.path.join(base_cfg.paths.output_dir, "optuna_search")
    os.makedirs(search_output_dir, exist_ok=True)

    prepare_func = DATASETS.get(base_cfg.general.dataset)
    if prepare_func is None:
        raise ValueError(f"поправь название датасета в конфиге: {base_cfg.general.dataset}")

    X_df, y_df, X_test_df, _, id_column_values = prepare_func(
        base_cfg.paths.train_csv, base_cfg.paths.test_csv,
        base_cfg.general.id_column, dummies=True, scale=False
    )
    X = X_df.values
    y = y_df.values.astype(np.float32)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=1),
    )
    study.optimize(
        lambda trial: objective(trial, base_cfg, device, X, y, search_output_dir),
        n_trials=n_trials,
    )

    print("Лучшие параметры:", study.best_params)
    print("Лучший RMSE (урезанный поиск):", study.best_value)


    best_cfg = OmegaConf.merge(base_cfg, {})
    best_cfg.model.params.hidden_dims = study.best_params["hidden_dims"]
    best_cfg.model.params.dropout_rate = study.best_params["dropout_rate"]
    best_cfg.training.lr = study.best_params["lr"]
    best_cfg.training.weight_decay = study.best_params["weight_decay"]
    best_cfg.training.batch_size = study.best_params["batch_size"]

    best_cfg.general.experiment_name = base_cfg.general.experiment_name + "_optuna_best"
    best_cfg.paths.output_dir = os.path.join(
        os.path.dirname(base_cfg.paths.output_dir), best_cfg.general.experiment_name
    )
    os.makedirs(best_cfg.paths.output_dir, exist_ok=True)

    best_config_path = os.path.join(best_cfg.paths.output_dir, "config.yaml")
    OmegaConf.save(config=best_cfg, f=best_config_path)
    print(f"Лучший конфиг сохранён: {best_config_path}")

    # финальное дообучение на лучших параметрах, полных n_splits/epochs
    oof_predictions = train(best_cfg, device, X, y)
    final_rmse = np.sqrt(np.mean((y - oof_predictions) ** 2))
    print(f"OOF rmse финальной модели: {final_rmse:.5f}")

    predict(best_cfg, device, X, X_test_df, id_column_values)


if __name__ == "__main__":
    main()