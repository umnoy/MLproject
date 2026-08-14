from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'optuna',
        'seed': 1,
        'task_type': 'regression',
        'dataset' : 'house_prices',
        'id_column': 'Id'
    },
    'paths': {
        'train_csv': './data/train.csv',
        'test_csv': './data/test.csv',
        'output_dir': './outputs/${general.experiment_name}',
    },
    'split': {
        'n_splits': 5,
    },
    'model': {
        'params': {
            'hidden_dims': [16, 8],
            'dropout_rate': 0.0,
        },
    },
    'training': {
        'device': 'cuda',
        'epochs': 50,
        'batch_size': 16,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'scheduler_epochs': 10,
        'scheduler_rate': 0.5,
    },
    'dataloader': {
        'num_workers': 0,
        'pin_memory': False,
    },
}

OmegaConf.save(config=OmegaConf.create(config), f='config.yaml')