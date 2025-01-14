from main import run_experiment
import os
from datetime import datetime

default_args = {
    'data': './data',
    'arch': 'simplecnn',
    'clustering': 'PCKmeans',
    'nmb_cluster': 10,
    'lr': 1e-2,
    'wd': -5,
    'reassign': 3.0,
    'workers': 4,
    'epochs': 10,
    'batch': 256,
    'momentum': 0.9,
    'resume': '',
    'checkpoints': 25000,
    'seed': 42,
    'exp': './experiment',
    'verbose': True,
    'device': 'mps',
    'plot_clusters': True,
    'label_fraction': 0.0,
    'label_pattern': 'random',
    'label_noise': 0.0,
    'cannot_link_fraction': 0.1,
    'must_link_fraction': 1.0,
    'pckmeans_iters': 5,
    'custom_clusters': None,
    'granularity': 1,
    'nmb_labeled_clusters': None,
    'violation_weight': 2,
}

experiments_sparsity_single = [
    {**default_args, 'label_fraction': 0.0},
]

# 1. overall sparsity

experiments_sparsity = [
    {**default_args, 'label_fraction': 0.0},
    {**default_args, 'label_fraction': 0.0005}, # 0.05%
    {**default_args, 'label_fraction': 0.001}, # 0.1%
    {**default_args, 'label_fraction': 0.002}, # 0.2%
    {**default_args, 'label_fraction': 0.005}, # 0.5%
    {**default_args, 'label_fraction': 0.01}, # 1%
]
# 2. overall sparsity w/ noise
# 	{0.005, 0.01 sparsity} x {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4 noise}
experiments_noise = [
    {**default_args, 'label_fraction': 0.005},
    {**default_args, 'label_fraction': 0.01},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.01},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.05},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.1},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.2},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.4},
    {**default_args, 'label_fraction': 0.005, 'label_noise': 0.6},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.01},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.05},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.1},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.2},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.4},
    {**default_args, 'label_fraction': 0.01, 'label_noise': 0.6},
]

# 3. class wise
# 	{1, 2, 5, 8 nmb_labeled_clusters} x {0.001, 0.01 label_fraction}
experiments_classwise = [
    {**default_args, 'nmb_labeled_clusters': 1, 'label_fraction': 0.001, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 1, 'label_fraction': 0.01, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 2, 'label_fraction': 0.001, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 2, 'label_fraction': 0.01, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 5, 'label_fraction': 0.001, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 5, 'label_fraction': 0.01, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 8, 'label_fraction': 0.001, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
    {**default_args, 'nmb_labeled_clusters': 8, 'label_fraction': 0.01, 'label_pattern': 'class_wise', 'cannot_link_fraction': 0.01},
]

# 4. granularity (must-link = 0, cannot-link = 0.01)
# 	vertical lines(1,4,7,9), circles(6,8,9,0) => 'custom_clusters' = [[1,4,7,9], [6,8,9,0]]
#   3, 5 superclasses => 'granularity' = 3, 2
experiments_granularity = [
    {**default_args, 'label_fraction': 0.01, 'must_link_fraction': 0.0, 'custom_clusters': [[1,4,7,9], [2,3,5,6,8,0]]},
    {**default_args, 'label_fraction': 0.01, 'must_link_fraction': 0.0, 'custom_clusters': [[6,8,9,0], [1,2,3,4,5,7]]},
    {**default_args, 'label_fraction': 0.01, 'must_link_fraction': 0.0, 'granularity': 5},
    {**default_args, 'label_fraction': 0.01, 'must_link_fraction': 0.0, 'granularity': 3},
    {**default_args, 'label_fraction': 0.01, 'must_link_fraction': 0.0, 'granularity': 2},
]

if __name__ == "__main__":
    # Run sparsity experiments
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = "sparsity_" + str(time)
    run_dir = os.path.join("./experiments", f"run_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for i, exp_cfg in enumerate(experiments_sparsity, start=1):
        config_dir = os.path.join(run_dir, f"config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        exp_cfg['exp'] = config_dir
        print(f"Running experiment in {config_dir} with config: {exp_cfg['label_fraction']}, {exp_cfg['label_noise']}")
        run_experiment(exp_cfg)

    # Run noise experiments
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = "noise_" + str(time)
    run_dir = os.path.join("./experiments", f"run_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for i, exp_cfg in enumerate(experiments_noise, start=1):
        config_dir = os.path.join(run_dir, f"config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        exp_cfg['exp'] = config_dir
        print(f"Running experiment in {config_dir} with config: {exp_cfg['label_fraction']}, {exp_cfg['label_noise']}")
        run_experiment(exp_cfg)

    # Run classwise experiments
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = "classwise_" + str(time)
    run_dir = os.path.join("./experiments", f"run_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for i, exp_cfg in enumerate(experiments_classwise, start=1):
        config_dir = os.path.join(run_dir, f"config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        exp_cfg['exp'] = config_dir
        print(f"Running experiment in {config_dir} with config: {exp_cfg['label_fraction']}, {exp_cfg['label_noise']}")
        run_experiment(exp_cfg)
    
    # Run granularity experiments
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = "granularity_" + str(time)
    run_dir = os.path.join("./experiments", f"run_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for i, exp_cfg in enumerate(experiments_granularity, start=1):
        config_dir = os.path.join(run_dir, f"config_{i}")
        os.makedirs(config_dir, exist_ok=True)
        exp_cfg['exp'] = config_dir
        print(f"Running experiment in {config_dir} with config: {exp_cfg['label_fraction']}, {exp_cfg['label_noise']}")
        run_experiment(exp_cfg)

    


