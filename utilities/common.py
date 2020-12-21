import numpy as np
import model_training.config as cfg
import json
import os


def flip_array(arr):
    unique_values = np.unique(arr)
    flipped = np.empty(np.array(arr).shape)
    for i in range(len(arr)):
        flipped[i] = np.flipud(unique_values)[np.where(unique_values == arr[i])]
    return flipped


def rgb_subtile(subtile):
    subtile_w0 = subtile.shape[0]
    subtile_w1 = subtile.shape[1]
    rgb_image = np.empty((subtile_w0, subtile_w1, 3))
    rgb_image[:, :, 0] = subtile[:, :, 3]
    rgb_image[:, :, 1] = subtile[:, :, 2]
    rgb_image[:, :, 2] = subtile[:, :, 0]
    return rgb_image


def make_training_log_file():
    # TODO: add dataset size etc, maybe dataset creation params?
    data = {
        'seed': cfg.seed,
        'input_shape': cfg.input_shape,
        'output_nodes': cfg.output_nodes,
        'dataset_size': cfg.dataset_size,
        'batch_size': cfg.batch_size,
        'epochs': cfg.epochs,
        'loss_function': cfg.loss_function,
        'lr': cfg.lr,
        'epsilon': cfg.epsilon,
        'beta1': cfg.beta1,
        'beta2': cfg.beta2,
        'depth_normalization': cfg.depth_normalization,
        'min_energy': cfg.min_energy,
        'max_energy': cfg.max_energy,
        'max_depth': cfg.max_depth
    }
    with open(os.path.join(cfg.output_dir, 'train_params.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_nan_mean(preds):
    return np.nanmean(np.dstack(preds), axis=2)
