import netCDF4 as nc
import os
import csv


fn_dir = '/home/mn/Downloads/forMahmoud/'
out_headers = ['lng', 'lat', 'z']
out_x = []
out_y = []
out_z = []
n_err = 0
n_good = 0
n_all = 0
n_dash = 0
for fn in os.listdir(fn_dir):
    if fn.endswith('.nc'):
        fn_path = os.path.join(fn_dir, fn)
        ncd = nc.Dataset(fn_path)
        print(ncd)

        n_x = len(ncd.variables['x'])
        n_y = len(ncd.variables['y'])
        n_k = len(ncd.variables['kKeep'])
        n_t = len(ncd.variables['time'])

        for i_t in range(n_t):
            for i_x in range(n_x):
                for i_y in range(n_y):
                    ncd_time = ncd.variables['time'][i_t]
                    ncd_x = ncd.variables['x'][i_x]
                    ncd_y = ncd.variables['y'][i_y]
                    z = None
                    for i_k in range(n_k):

                        ncd_kKeep = ncd.variables['kKeep'][i_k]
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
                    if n_all % 1000 == 0:
                        print(f'all: {n_all}, keep: {n_good}, errs: {n_err}, dash: {n_dash}')

        print(f'Tile: {fn}')
        print(f'    Total: {n_all}, 1k: {n_good}, nk: {n_err}, --: {n_dash}')
        print(f'    len(x): {len(out_x)}, len(y): {len(out_y)}, len(z): {len(out_z)}')

        print(f'    Creating CSV file for {fn}...')
        with open(f'out_{fn}.xyz', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(out_headers)
            writer.writerows(zip(out_x, out_y, out_z))
            print(f'    out_{fn}.xyz')

        out_x = []
        out_y = []
        out_z = []

print('Finished')
