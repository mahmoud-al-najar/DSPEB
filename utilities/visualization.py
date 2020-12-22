import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_bathy(bathy, show=True):
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 12})
    np.set_printoptions(suppress=True)
    plt.scatter(bathy.lng, bathy.lat, c=bathy.z, cmap='ocean_r', vmax=50, marker=',', s=np.full(len(bathy.lng), 1))
    plt.colorbar(shrink=0.5)
    plt.tight_layout(pad=2)
    plt.suptitle(bathy.id)
    if show:
        plt.show()


def visualize_bathy_and_satellite(target_bathy, cropped_sentinel_image):
    plt.rcParams.update({'font.size': 12})
    np.set_printoptions(suppress=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(cropped_sentinel_image, extent=[target_bathy.west, target_bathy.east,
                                                               target_bathy.south, target_bathy.north])
    im = ax.scatter(target_bathy.lng, target_bathy.lat, c=target_bathy.z, cmap='ocean_r', vmin=np.min(target_bathy.z),
                    vmax=np.max(target_bathy.z), marker=',', s=np.full(len(target_bathy.lng), 1))

    for tick in ax.get_xticklabels():
        tick.set_rotation(-45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical', shrink=0.5)
    plt.tight_layout()
    plt.show()
