import os
import time
import gdal
import shutil
import numpy as np
import pandas as pd
import datagen_config as cfg
from utilities.common import get_blue_ratio
from utilities.data_io import make_tarfile
from utilities.data_io import parse_sentinel2_tiles_metadata, get_bathy_xyz
from utilities.preprocessing import apply_fft, apply_per_band_min_max_normalization, apply_normxcorr2

# get list of Sentinel-2 Tile objects
sentinel2tile_list = parse_sentinel2_tiles_metadata()
print(sentinel2tile_list)
print(f'len(tiles) = {len(sentinel2tile_list)}')

# get bathymetry xyz
bathy_xyz = get_bathy_xyz(sentinel2tile_list)
x, y, z = bathy_xyz
print(len(x), len(y), len(z))
print(x)

if not os.path.exists(cfg.out_path_dataset):
    os.mkdir(cfg.out_path_dataset)

if not os.path.exists(cfg.out_path_tmpdir):
    os.mkdir(cfg.out_path_tmpdir)

tmp_dirname = os.path.join(cfg.out_path_tmpdir, f'tmp_{cfg.region}')
if not os.path.exists(tmp_dirname):
    os.mkdir(tmp_dirname)

nb_tiles = len(cfg.tiles)

print(f'####################################### Bathy Loading ################################################')

x, y, z = bathy_xyz

for i in range(nb_tiles):
    print(f'Tile {cfg.tiles[i]}: {len(x[i])} measurement points')

good = 0
bad1 = 0  # too close to the tile border
bad2 = 0  # nan or inf
bad3 = 0  # ValueError
bad4 = 0  # Clouds
dataframe = pd.DataFrame([], columns=['z', 'x', 'y', 'epsg', 'max_energy'])

for i in range(len(sentinel2tile_list)):
    tile = sentinel2tile_list[i]
    print(f'Tile : {tile.id}')
    for safe in tile.safes:
        nb1 = 0
        t = time.time()
        a = os.listdir(os.path.join(safe.s2_path, 'GRANULE'))
        path = os.path.join(safe.s2_path, 'GRANULE', a[0], 'IMG_DATA', f'T{tile.id}_{safe.date}T{safe.time}_')

        for k in range(len(x[i])):
            z_tid = z[i][k] + safe.tidal_elevation
            if cfg.depth_lim_min <= z_tid <= cfg.depth_lim_max:
                cx = int((x[i][k] - tile.corner['x']) / 10 - cfg.w_sub_tile / 2)
                cy = int((tile.corner['y'] - y[i][k]) / 10 + cfg.w_sub_tile / 2)
                if cx + cfg.w_sub_tile < cfg.w_sentinel / 10 - 1 and \
                        cx > 0 and \
                        cy + cfg.w_sub_tile < cfg.w_sentinel / 10 - 1 and \
                        cy > 0:
                    nb1 += 1

                    Bands = np.empty((cfg.w_sub_tile, cfg.w_sub_tile, 4))
                    Bands[:, :, 0] = np.array(
                        gdal.Open(path + 'B02.jp2').ReadAsArray(cx, cy, cfg.w_sub_tile, cfg.w_sub_tile)) / 4096
                    Bands[:, :, 2] = np.array(
                        gdal.Open(path + 'B03.jp2').ReadAsArray(cx, cy, cfg.w_sub_tile, cfg.w_sub_tile)) / 4096
                    Bands[:, :, 3] = np.array(
                        gdal.Open(path + 'B04.jp2').ReadAsArray(cx, cy, cfg.w_sub_tile, cfg.w_sub_tile)) / 4096
                    Bands[:, :, 1] = np.array(
                        gdal.Open(path + 'B08.jp2').ReadAsArray(cx, cy, cfg.w_sub_tile, cfg.w_sub_tile)) / 4096

                    if np.isnan(np.min(Bands)) or np.isinf(np.min(Bands)):
                        bad2 += 1
                    else:
                        try:
                            ratio_blue = get_blue_ratio(Bands)
                            if ratio_blue < 0.8:
                                bad4 += 1
                            else:

                                if apply_fft in cfg.preprocessing_funcs:
                                    B_fft, flag, max_energy = apply_fft(Bands,
                                                                    energy_min_thresh=cfg.min_energy,
                                                                    energy_max_thresh=cfg.max_energy)
                                if apply_normxcorr2 in cfg.preprocessing_funcs:
                                    B_fnxc = apply_normxcorr2(B_fft)
                                if apply_per_band_min_max_normalization in cfg.preprocessing_funcs:
                                    B_fnxc = apply_per_band_min_max_normalization(B_fnxc)

                                good += 1
                                num = '{0:05}'.format(good)

                                tmp_name = f'{tmp_dirname}/{num}_{tile.id}_{safe.date}'
                                np.save(tmp_name, B_fnxc)

                                tmp_name_raw = f'{tmp_name}_RAW'
                                np.save(tmp_name_raw, Bands)

                                df = pd.DataFrame(
                                    [[z_tid, x[i][k], y[i][k], z[i][k], tile.epsgs[0], max_energy]],
                                    index=[tmp_name],
                                    columns=['z', 'x', 'y', 'z_no_tide', 'epsg', 'max_energy'])
                                dataframe = dataframe.append(df)
                                if len(dataframe.index) % 1000 == 0:
                                    dataframe.to_csv(
                                        cfg.out_path_csv + cfg.region + '_' + str(len(dataframe.index)) + '_.csv')
                                    tmp_tarname = cfg.out_path_tmpdir + '/' + cfg.region + '_' + str(
                                        len(dataframe.index)) + '_TEMP.tar.gz'
                                    make_tarfile(tmp_tarname, tmp_dirname)
                                    shutil.copy(tmp_tarname, cfg.out_path_dataset)

                        except ValueError:
                            bad3 += 1

        print(f'       {nb1} subtiles computed in {tile.id} {safe.date}')
        print(f'       Computational time : {time.time() - t}')

dataframe.to_csv(os.path.join(cfg.out_path_csv, f'{cfg.region}_FULL.csv'))
tmp_tarname = os.path.join(cfg.out_path_tmpdir, f'{cfg.region}_{str(len(dataframe.index))}_FULL.tar.gz')
make_tarfile(tmp_tarname, tmp_dirname)
shutil.copy(tmp_tarname, cfg.out_path_dataset)

print('###################################################################################################')
print(f'{good + bad1 + bad2 + bad3 + bad4} Input samples')
print(f'{good} Good samples')
print(f'{bad1 + bad2 + bad3 + bad4} Rejected samples, ({bad1}, {bad2}, {bad3}, {bad4})')
print(f'##################################################################################################')
