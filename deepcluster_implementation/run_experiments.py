from main import run_experiment
import os
from datetime import datetime

default_args = {
    'data': './data',
    'arch': 'simplecnn',
    'sobel': False,
    'clustering': 'PCKmeans',
    'nmb_cluster': 10,
    'lr': 5e-2,
    'wd': -5,
    'reassign': 3.0,
    'workers': 4,
    'epochs': 10,
    'batch': 256,
    'momentum': 0.9,
    'resume': '',
    'checkpoints': 25000,
    'seed': 31,
    'exp': './experiment',
    'verbose': True,
    'device': 'mps',
    'plot_clusters': True,
    'label_fraction': 0.0,
    'label_pattern': 'random',
    'label_noise': 0.0,
    'cannot_link_fraction': 0.1
}

experiments = [
    {**default_args},
    {**default_args, 'label_fraction': 0.001, 'label_noise': 0.0},
    # {**default_args, 'label_fraction': 0.01, 'label_noise': 0.0},
    # {**default_args, 'label_fraction': 0.1, 'label_noise': 0.0},
    # {**default_args, 'label_fraction': 0.001, 'label_noise': 0.1},
    # {**default_args, 'label_fraction': 0.01, 'label_noise': 0.1},
    # {**default_args, 'label_fraction': 0.1, 'label_noise': 0.1}
]

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(default_args['exp'], timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    default_args['exp'] = experiment_dir
    for exp_cfg in experiments:
        print(f"Running experiment with config: {exp_cfg}")
        run_experiment(exp_cfg)
