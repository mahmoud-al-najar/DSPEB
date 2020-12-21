import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from utilities.common import read_nc_file


fn_dir = '/media/mn/WD4TB/PEPS/data/pbm_results/22NCL'
dir_out = '/media/mn/WD4TB/PEPS/data/pbm_results/22NCL'
out_headers = ['lng', 'lat', 'z']
show = True
save_fig = True
for fn in os.listdir(fn_dir):
    if fn.endswith('.nc'):
        fn_path = os.path.join(fn_dir, fn)
        lng, lat, z = read_nc_file(fn_path)

        if show or save_fig:
            plt.figure(figsize=(8, 5))
            plt.rcParams.update({'font.size': 12})
            np.set_printoptions(suppress=True)
            plt.scatter(lng, lat, c=z, cmap='ocean_r', vmax=50, marker=',', s=np.full(len(lng), 1))
            plt.colorbar(shrink=0.5)
            plt.tight_layout(pad=2)
            plt.suptitle(f'{fn.replace(".nc", "")}')
            if show:
                plt.show()
            if save_fig:
                plt.savefig(os.path.join(dir_out, f'{fn}.png'))
            plt.close()

        with open(os.path.join(dir_out, f'{fn}.xyz'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(out_headers)
            writer.writerows(zip(lng, lat, z))
            print(f'    out_{fn}.xyz')
