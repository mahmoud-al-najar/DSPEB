import os
import json
import numpy as np
import model_training.config as train_cfg
import datagen_config as datagen_cfg

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
        'seed': train_cfg.seed,
        'input_shape': train_cfg.input_shape,
        'output_nodes': train_cfg.output_nodes,
        'dataset_size': train_cfg.dataset_size,
        'batch_size': train_cfg.batch_size,
        'epochs': train_cfg.epochs,
        'loss_function': train_cfg.loss_function,
        'lr': train_cfg.lr,
        'epsilon': train_cfg.epsilon,
        'beta1': train_cfg.beta1,
        'beta2': train_cfg.beta2,
        'depth_normalization': train_cfg.depth_normalization,
        'min_energy': train_cfg.min_energy,
        'max_energy': train_cfg.max_energy,
        'max_depth': train_cfg.max_depth
    }
    with open(os.path.join(train_cfg.output_dir, 'train_params.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def make_datagen_log_file():
    data = {
        'in_path_bathy': datagen_cfg.in_path_bathy,
        'in_path_s2': datagen_cfg.in_path_s2,
        'in_path_tidal': datagen_cfg.in_path_tidal,
        'out_path_dir': datagen_cfg.out_path_dir,
        'out_path_tmpdir': datagen_cfg.out_path_tmpdir,
        'region': datagen_cfg.region,
        'tiles': datagen_cfg.tiles,
        'w_sub_tile': datagen_cfg.w_sub_tile,
        'w_sentinel': datagen_cfg.w_sentinel,
        'min_energy': datagen_cfg.min_energy,
        'max_energy': datagen_cfg.max_energy,
        'max_cc': datagen_cfg.max_cc,
        'nb_max_date': datagen_cfg.nb_max_date,
        'depth_lim_min': datagen_cfg.depth_lim_min,
        'depth_lim_max': datagen_cfg.depth_lim_max,
        'nb_max_pt_per_tile': datagen_cfg.nb_max_pt_per_tile,
        'line_max_read': datagen_cfg.line_max_read,
        'out_path_csv': datagen_cfg.out_path_csv,
        'out_path_dataset': datagen_cfg.out_path_dataset,
        'preprocessing_funcs': datagen_cfg.preprocessing_funcs
    }
    with open(os.path.join(datagen_cfg.out_path_csv, 'datagen_params.json'), 'w', encoding='utf-8') as f:
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

    cx = int((x_point - x_corner) - train_cfg.w_sub_tile * 5)
    cy = int((y_corner - y_point) - train_cfg.w_sub_tile * 5)

    return (cx > 0) and \
           (cx < (train_cfg.w_sentinel - train_cfg.w_sub_tile * 10)) and \
           (cy > 0) and \
           (cy < (train_cfg.w_sentinel - train_cfg.w_sub_tile * 10))
