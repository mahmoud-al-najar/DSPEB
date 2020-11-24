import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import data_generation.config as cfg


fig, axes = plt.subplots(1, 5, figsize=(20, 6))
im_ax = 0
for f in os.listdir('../temporary'):
    if f.endswith('.xyz'):
        df = pd.read_csv(os.path.join('../temporary', f))
        print(f)
        im = axes[im_ax].scatter(df.lng, df.lat, c=df.z, cmap='ocean_r', vmax=40)
        for tick in axes[im_ax].get_xticklabels():
            tick.set_rotation(-45)
        im_ax += 1


divider = make_axes_locatable(axes[len(axes) - 1])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()
