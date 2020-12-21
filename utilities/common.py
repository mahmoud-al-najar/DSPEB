import os
import json
import numpy as np
import model_training.config as cfg

from utilities.preprocessing import apply_per_band_min_max_normalization


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


def get_blue_ratio(sub_tile, point_name='P-Name'):
    nx, ny, nc = sub_tile.shape
    count_all_values = nx * ny

    raw_rgb_subtile = np.empty((nx, ny, 3))
    raw_rgb_subtile[:, :, 0] = sub_tile[:, :, 3]
    raw_rgb_subtile[:, :, 1] = sub_tile[:, :, 2]
    raw_rgb_subtile[:, :, 2] = sub_tile[:, :, 0]

    normalized_rgb_subtile = np.empty((nx, ny, 3))
    normalized_rgb_subtile[:, :, 0] = raw_rgb_subtile[:, :, 0]
    normalized_rgb_subtile[:, :, 1] = raw_rgb_subtile[:, :, 1]
    normalized_rgb_subtile[:, :, 2] = raw_rgb_subtile[:, :, 2]

    normalized_rgb_subtile = apply_per_band_min_max_normalization(normalized_rgb_subtile)
    median = np.nanmedian(normalized_rgb_subtile)

    n_blue_pixels = 0
    if median < 0.15:
        for x in range(nx):
            for y in range(ny):
                r, g, b = normalized_rgb_subtile[x, y, :]
                if r < 0.2 and g < 0.2 and b < 0.2:
                    n_blue_pixels += 1
        return n_blue_pixels / count_all_values
    else:
        for x in range(nx):
            for y in range(ny):
                r, g, b = raw_rgb_subtile[x, y, :]
                if (r < 0.2 < b and g < b) or (g < 0.2 < b and r < b):
                    n_blue_pixels += 1
        return n_blue_pixels / count_all_values


def isin_tile(x_point, y_point, x_corner, y_corner):
    """
    This function check if the point is in the sentinel tile
    according to its top left corner (lm*lm)
    :param x_point: x coordinate of the given point
    :param y_point: y coordinate of the given point
    :param x_corner: x coordinate of the top left corner of the given tile
    :param y_corner: y coordinate of the top left corner of the given tile
    :return: boolean, true if in, else false
    """

    cx = int((x_point - x_corner) - cfg.w_sub_tile * 5)
    cy = int((y_corner - y_point) - cfg.w_sub_tile * 5)

    return (cx > 0) and \
          (cx < (cfg.w_sentinel - cfg.w_sub_tile * 10)) and \
          (cy > 0) and \
          (cy < (cfg.w_sentinel - cfg.w_sub_tile * 10))
