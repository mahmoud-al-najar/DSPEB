import numpy as np
import model_training.config as cfg
import json
import os
import netCDF4 as nc


def __flip_bathymetry_y_axis(arr):
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


def read_nc_file(path_to_nc, projection_in=None, projection_out=None):
    ncd = nc.Dataset(path_to_nc)
    print(ncd)

    n_x = len(ncd.variables['x'])
    n_y = len(ncd.variables['y'])
    n_k = len(ncd.variables['kKeep'])
    n_t = len(ncd.variables['time'])

    out_x = []
    out_y = []
    out_z = []
    n_err = 0
    n_good = 0
    n_all = 0
    n_dash = 0

    for i_t in range(n_t):
        for i_x in range(n_x):
            for i_y in range(n_y):
                ncd_time = ncd.variables['time'][i_t]
                ncd_x = ncd.variables['x'][i_x]
                ncd_y = ncd.variables['y'][i_y]
                z = None
                for i_k in range(n_k):
                    ncd_z = ncd['depth'][i_y, i_x, i_k, i_t]
                    n_all += 1

                    if ncd_z != '--':
                        if z is None:
                            z = ncd_z
                            n_good += 1
                            out_x.append(ncd_x)
                            out_y.append(ncd_y)
                            out_z.append(z)
                        else:
                            new_z = (z + ncd_z) / 2
                            z = new_z
                            n_err += 1
                    else:
                        n_dash += 1
                if n_all % 5000 == 0:
                    print(f'all: {n_all}, keep: {n_good}, errs: {n_err}, dash: {n_dash}')
    fn = path_to_nc.split("/")[-1]
    print(f'Filename: {fn}')
    print(f'    Total: {n_all}, 1k: {n_good}, nk: {n_err}, --: {n_dash}')
    print(f'    len(x): {len(out_x)}, len(y): {len(out_y)}, len(z): {len(out_z)}')

    print(f'    Creating CSV file for {fn}...')
    out_y = __flip_bathymetry_y_axis(out_y)
    return out_x, out_y, out_z
