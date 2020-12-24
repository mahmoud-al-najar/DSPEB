import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from utilities.common import get_rgb_subtile, get_blue_ratio
from utilities.visualization import visualize_bathy_and_satellite
from utilities.wrappers import Sentinel2Safe, BathymetryXYZ
from utilities.preprocessing import apply_fft, apply_normxcorr2


def reconstruct_safe(model, tile_id, path_safe, target_bathy, out_resolution=100, subtile_l=400):
    safe = Sentinel2Safe(safe_path=path_safe, tile_id=tile_id)
    # cropped_sentinel_image = safe.get_image_around_bathy(target_bathy)
    # visualize_bathy_and_satellite(target_bathy, get_rgb_subtile(cropped_sentinel_image))

    w = int(target_bathy.width/out_resolution)  # raw difference to array length
    h = int(target_bathy.height/out_resolution)
    tb_north, tb_south, tb_east, tb_west = target_bathy.north, target_bathy.south, \
                               target_bathy.east, target_bathy.west

    reconstructed_bathy = BathymetryXYZ(path=None)
    # headers = ('lng', 'lat', 'z')
    # reconstructed_bathy.append(headers)
    for y in range(h):
        print(f'{y+1}/{h}')
        batch_images = []
        batch_coordinates = []
        batch_energies = []
        batch_blue_ratios = []
        for x in range(w):
            north = tb_north - (y * out_resolution)
            south = north - subtile_l
            west = tb_west + (x * out_resolution)
            east = west + subtile_l
            subtile = safe.get_subtile_between_coordinates(north, south, east, west)
            blue_ratio = get_blue_ratio(subtile)
            subtile, flag, max_energy = apply_fft(subtile, energy_min_thresh=None,
                                                energy_max_thresh=None)
            # if max_energy > 0.01:
            subtile = apply_normxcorr2(subtile)

            center_lng = east - (subtile_l / 2)
            center_lat = north - (subtile_l / 2)
            coords = (center_lng, center_lat)

            batch_images.append(subtile)
            batch_coordinates.append(coords)
            batch_energies.append(max_energy)
            batch_blue_ratios.append(blue_ratio)

        if len(batch_images) > 0:
            batch = np.empty((len(batch_images), int(subtile_l/10), int(subtile_l/10), 4))
            for i in range(len(batch_images)):
                batch[i, :, :, :] = batch_images[i]
            batch_results = model.predict(batch) * 10
            batch_results = batch_results[:, 0] - safe.tidal_elevation
            for i in range(len(batch_results)):
                res_dep = batch_results[i]
                coords = batch_coordinates[i]
                energy = batch_energies[i]
                blue_ratio = batch_blue_ratios[i]
                res_lng = coords[0]
                res_lat = coords[1]
                reconstructed_bathy.lng.append(res_lng)
                reconstructed_bathy.lat.append(res_lat)
                reconstructed_bathy.z.append(res_dep)
                reconstructed_bathy.energies.append(energy)
                reconstructed_bathy.blue_ratios.append(blue_ratio)

    return reconstructed_bathy


path_target_bathy_xyz = '/media/mn/WD4TB/PEPS/saint_louis_data/shom/saint_louis/saint_louis/saint_louis.fxyz'
tile_id = 'T28PCC'
path_safe_to_reconstruct = '/media/mn/WD4TB/PEPS/saint_louis_data/sentinel2/saint_louis/28PCC/S2B_MSIL1C_20180128T113329_N0206_R080_T28PCC_20180128T145508.SAFE'
path_trained_model = '/media/mn/WD4TB/PEPS/mixed_training/nonorm/fixed/' \
                     'ResNet56_16981_lr_0.0001__epsilon_1e-08__beta1_0.99__beta2_0.999/ResNet56_16981_66_0.29.hdf5'

model = load_model(path_trained_model)
target_bathy = BathymetryXYZ(path_target_bathy_xyz)
resb = reconstruct_safe(model, tile_id, path_safe_to_reconstruct, target_bathy, out_resolution=400)
resb.write_xyz_file('resb.xyz')

safe = Sentinel2Safe(safe_path=path_safe_to_reconstruct, tile_id=tile_id)
visualize_bathy_and_satellite(resb, get_rgb_subtile(safe.get_image_around_bathy(resb)))
