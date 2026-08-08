from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'mlp_baseline2',
        'seed': 42,
        'task_type': 'binary',
        'dataset' : 'titanic',
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
            'hidden_dims': [64, 32],
            'dropout_rate': 0.3,
        },
    },
    'training': {
        'device': 'cuda',
        'epochs': 50,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-4,
    },
    'dataloader': {
        'num_workers': 0,
        'pin_memory': False,
    },
}

OmegaConf.save(config=OmegaConf.create(config), f='config.yaml')