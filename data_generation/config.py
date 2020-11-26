import os
import sys

in_path_bathy = f'bathy'
in_path_s2 = 'sentinel2'
in_path_tidal = 'S2_ERA5_metaAndEnvironmentalData.nc'

out_path_dir = 'job_output'

#out_path_tmpdir = os.path.join(out_path_dir, 'tmpdir')
out_path_tmpdir = str(sys.argv[1])

region = 'guyane'
tiles = ['21NZG']

w_sub_tile = 40
w_sentinel = 109_800

min_energy = 0.01
max_energy = 3
max_cc = 5
nb_max_date = 7
depth_lim_min = 0
depth_lim_max = 120
nb_max_pt_per_tile = 1500
line_max_read = 10_000

verbose = 0

in_path_bathy = f'{os.path.join(in_path_bathy, region)}'
in_path_s2 = os.path.join(in_path_s2, region)
out_path_csv = os.path.join(out_path_dir, 'CSV')  # , f'{region}.csv'
out_path_dataset = os.path.join(out_path_dir, f'dataset_{region}')
