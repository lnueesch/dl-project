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
    # Create a run folder under ./experiments
    run_dir = os.path.join("./experiments", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    for i, exp_cfg in enumerate(experiments, start=1):
        config_dir = os.path.join(run_dir, f"config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        exp_cfg['exp'] = config_dir  # pass the config folder to run_experiment
        print(f"Running experiment in {config_dir} with config: {exp_cfg}")
        run_experiment(exp_cfg)
