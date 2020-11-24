import os


# in_path_bathy = f'/media/mn/WD4TB/PEPS/data/bathy/'
# in_path_s2 = '/media/mn/WD4TB/PEPS/data/capbreton_validation/'
in_path_bathy = f'/media/mn/WD4TB/PEPS/saint_louis_data/pmb_bathymetry_results'
in_path_s2 = '/media/mn/WD4TB/PEPS/saint_louis_data/sentinel2'
in_path_tidal = '/media/mn/WD4TB/PEPS/data/S2_ERA5_metaAndEnvironmentalData.nc'

out_path_dir = 'outs'

out_path_tmpdir = os.path.join(out_path_dir, 'tmpdir')
# out_path_tmpdir = str(sys.argv[1])

region = 'saint_louis'
tiles = ['28PCC']

w_sub_tile = 80
w_sentinel = 109_800

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
