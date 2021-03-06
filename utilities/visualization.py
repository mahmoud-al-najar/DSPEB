import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from utilities.data_io import get_top_left_corner_coordinates_for_image


def visualize_bathy_xyz(lng, lat, z, title=None, show=True, vmax=50):
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 12})
    np.set_printoptions(suppress=True)
    plt.scatter(lng, lat, c=z, cmap='ocean_r', vmax=vmax, marker=',', s=np.full(len(lng), 1))
    plt.colorbar(shrink=0.5)
    plt.tight_layout(pad=2)
    if title:
        plt.suptitle(title)
    if show:
        plt.show()


def visualize_bathy(bathy, show=True):
    visualize_bathy_xyz(lng=bathy.lng, lat=bathy.lat, z=bathy.z, title=bathy.id, show=show)


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


def visualize_bathy_and_full_satellite(target_bathy, safe):
    plt.rcParams.update({'font.size': 12})
    np.set_printoptions(suppress=True)
    sentinel_image = safe.get_full_rgb_image()
    x_corner, y_corner, epsg = get_top_left_corner_coordinates_for_image(safe.s2_path)
    xc = x_corner
    yc = y_corner
    x_max = xc + 109_800
    y_min = yc - 109_800

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(sentinel_image, extent=[xc, x_max, y_min, yc])
    im = ax.scatter(target_bathy.lng, target_bathy.lat, c=target_bathy.z, cmap='ocean_r', vmin=np.min(target_bathy.z),
                    vmax=np.max(target_bathy.z), marker=',', s=np.full(len(target_bathy.lng), 1))

    for tick in ax.get_xticklabels():
        tick.set_rotation(-45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical', shrink=0.5)
    plt.tight_layout()
    plt.show()
