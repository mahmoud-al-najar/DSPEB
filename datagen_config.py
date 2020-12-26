import os
import sys
from utilities.preprocessing import apply_fft, apply_normxcorr2, apply_per_band_min_max_normalization

in_path_bathy = f'bathy'
in_path_s2 = 'sentinel2'
in_path_tidal = '/media/mn/WD4TB/PEPS/data/S2_ERA5_metaAndEnvironmentalData.nc'
in_path_safes_list = 'safes.txt'

out_path_dir = 'job_output'

out_path_tmpdir = os.path.join(out_path_dir, 'tmpdir')
# out_path_tmpdir = str(sys.argv[1])

region = 'guyane'
tiles = ['21NZG']

w_sub_tile = 40
w_sentinel = 109_800

min_energy = 0.01  # NO EFFECT, all subtiles are saved currently
max_energy = 3  # NO EFFECT, all subtiles are saved currently
max_cc = 5
nb_max_date = 7
depth_lim_min = 0
depth_lim_max = 120
nb_max_pt_per_tile = 1500
line_max_read = 50_000

verbose = 0

in_path_bathy = f'{os.path.join(in_path_bathy, region)}'
in_path_s2 = os.path.join(in_path_s2, region)
in_path_datalake_s2 = '/datalake/S2-L1C'
out_path_csv = os.path.join(out_path_dir, 'CSV')
out_path_dataset = os.path.join(out_path_dir, f'dataset_{region}')

preprocessing_funcs = [apply_fft, apply_normxcorr2]
