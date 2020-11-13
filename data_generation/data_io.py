import xml.etree.ElementTree as et
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import os
import time
import tarfile
import data_generation.config as cfg
from data_generation.sentinel2wrappers import Sentinel2Tile, Sentinel2Safe
from data_generation.utils import isin_tile
import pyproj
import pandas as pd
import warnings


def get_cloud_coverage(path):
    """
    Find the cloud coverage of the S2-L1C image in the xml file
    :param path:(str) path to the .SAFE repository of the image
    :return: cloud_coverage (float)
    """
    # xml = path + 'MTD_MSIL1C.xml'
    xml = os.path.join(path, 'MTD_MSIL1C.xml')
    tree = et.parse(xml)
    root = tree.getroot()
    cloud_coverage = float(root[3][0].text)
    return cloud_coverage


def get_top_left_corner_coordinates_for_image(path):
    """
    Find the x, y coordinates of the top left corner of the S2-L1C image in the WGS84 coordinate system of the tile
    and its epsg reference number
    :param path:(str) path to the .SAFE repository of the image
    :return: x_corner, y_corner, epsg (int, int, str)
    """

    # xml = path + 'GRANULE/' + os.listdir(path + 'GRANULE/')[0] + '/MTD_TL.xml'
    xml = os.path.join(path, 'GRANULE', os.listdir(os.path.join(path, 'GRANULE'))[0], 'MTD_TL.xml')
    tree = et.parse(xml)
    root = tree.getroot()
    x_corner = int(root[1][0][5][0].text)
    y_corner = int(root[1][0][5][1].text)
    epsg = root[1][0][1].text
    return x_corner, y_corner, epsg


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def get_tidal_elevation_for_image(safe):
    date = safe[11:19]
    time = safe[20:26]
    tidal_path = cfg.in_path_tidal
    df = xr.open_dataset(tidal_path).to_dataframe()
    tid = df[df['S2_fname'] == safe[0:len(safe) - 5]]

    if tid['prediction_at_ERA5_point'].empty:
        print(f'     {date} - {time}: no tidal elevation data')
        return None
    else:
        if np.isnan(tid['prediction_at_ERA5_point'].values[0]):
            delta_t = timedelta(hours=1)
            timing = datetime.strptime(date + time, '%Y%m%d%H%M%S')
            timing_start = timing - delta_t
            timing_end = timing + delta_t
            timing_start = timing_start.strftime('%Y-%m-%d %H:%M:%S')
            timing_end = timing_end.strftime('%Y-%m-%d %H:%M:%S')
            mask = (df['S2_time'] > timing_start) & (df['S2_time'] < timing_end)
            timing_search = df.loc[mask]
            b = 1  # +/- 1 degree to search for the tidal information (around the actual tile center)
            a = timing_search[(timing_search['S2_lon'] < (tid['S2_lon'].values[0] + b)) &
                              (timing_search['S2_lon'] > (tid['S2_lon'].values[0] - b)) &
                              (timing_search['S2_lat'] < (tid['S2_lat'].values[0] + b)) &
                              (timing_search['S2_lat'] > (tid['S2_lat'].values[0] - b))]
            tidal = np.nanmean(a['prediction_at_ERA5_point'].values)
            # print(f'     {date} - {time} : tidal elevation = {"{0:0.3f}".format(tidal)}m')
            return tidal
        else:
            tidal = tid['prediction_at_ERA5_point'].values[0]
            # if cfg.verbose >= 0:
                # print(f'     {date} - {time} : tidal elevation = {"{0:0.3f}".format(tidal)}m')
            return tidal


# TODO: refactor sentinel2wrappers.py. pass path to constructors then parse all metadata from there
def parse_sentinel2_tiles_metadata():
    """
            This function returns the info of the nb_max_date tiles with the smallest cloud coverage
            :return: corners, paths, dates, epsgs: infos of the selected tiles
            """
    sentinel2_tiles = []
    i = -1
    for tile in cfg.tiles:
        temp_tile = Sentinel2Tile()
        temp_tile.id = tile
        n = 0
        i += 1  # tile index
        if cfg.verbose >= 0:
            print(f'{tile}')
        path_t = os.path.join(cfg.in_path_s2, tile)
        safes = os.listdir(path_t)
        temp = []
        for safe in safes:
            if safe.endswith('SAFE'):
                temp.append(safe)
        safes = temp
        for safe in safes:
            temp_safe = Sentinel2Safe()
            path_s = os.path.join(path_t, safe)
            cloud_coverage = "{0:0.2f}".format(get_cloud_coverage(path_s))
            if float(cloud_coverage) < cfg.max_cc and n < cfg.nb_max_date:
                date = safe[11:19]
                time = safe[20:26]
                tidal = get_tidal_elevation_for_image(safe)
                if tidal:
                    temp_safe.tidal_elevation = tidal
                    temp_safe.date = date
                    temp_safe.time = time
                    temp_safe.s2_path = path_s
                    x, y, epsg = get_top_left_corner_coordinates_for_image(path_s)
                    print(x, y, epsg, path_s)
                    temp_safe.corners = (x, y)

                    temp_safe.epsg = epsg
                    temp_tile.safes.append(temp_safe)

                    if epsg not in temp_tile.epsgs:
                        temp_tile.epsgs.append(epsg)
                        if len(temp_tile.epsgs) > 1:
                            warnings.warn(f'==================================== Tile {temp_tile.id}: multiple epsg\'s')
                            exit()
                    if temp_tile.corner['x'] and temp_tile.corner['y']:
                        if x != temp_tile.corner['x'] or y != temp_tile.corner['y']:
                            id = temp_tile.id
                            warnings.warn(
                                f'============================================ Tile {id}: multiple corners')
                            exit()
                        else:
                            pass  # Different snapshots of the same tile should have the exact same corners
                    else:
                        temp_tile.corner['x'] = x
                        temp_tile.corner['y'] = y

                n += 1
        sentinel2_tiles.append(temp_tile)

    return sentinel2_tiles


def get_bathy_xyz(sentinel2tile_list):
    """
    This function returns the useful bathy points according to 2 criteria :
    distance to each others (not to much redundancy) & not to close to tile borders & depth limited (+ & -)
    :param path: path of the bathymetry
    :param x: x coordinates of already kept bathy points
    :param y: y coordinates of already kept bathy points
    :param z: z coordinates of already kept bathy points
    :return: (x, y, z): coordinates of the bathy points kept, appended to the previous ones
    """

    precision = 10
    nb_tiles = len(cfg.tiles)

    # first bathy pts filtering
    x, y, z = [[]] * nb_tiles, [[]] * nb_tiles, [[]] * nb_tiles

    proj = [[]] * nb_tiles
    for i in range(nb_tiles):
        tile = sentinel2tile_list[i]
        if len(tile.epsgs) == 1:
            proj[i] = pyproj.Proj(proj='utm', init=tile.epsgs[0], ellps='WGS84')
        elif len(tile.epsgs) > 1:
            warnings.warn(f'AGAIN =================================================== Tile {tile.id}: multiple epsg\'s')
            exit()
        else:
            warnings.warn(
                f'THIS SHOULD NEVER HAPPEN ====================================== didn\nt find tile {tile.id}\'s epsg?')
            exit()

    bathy_points = pd.DataFrame()
    for directory in os.listdir(cfg.in_path_bathy):
        path = f'{os.path.join(cfg.in_path_bathy, directory, directory)}.xyz'
        df = pd.read_csv(path, header=2)
        bathy_points = bathy_points.append(df, ignore_index=True)

    bins = np.linspace(cfg.depth_lim_min, 100, 10)
    max_depth = bathy_points['depth(m - down positive - LAT)'].max()
    if max_depth > 100:
        bins = np.append(bins, max_depth)
    labels = np.linspace(0, len(bins) - 2, len(bins) - 1)
    bathy_points['depth label'] = pd.cut(bathy_points['depth(m - down positive - LAT)'], bins=bins, labels=labels)
    print(bathy_points['depth label'].value_counts(sort=False))

    n_tot = 0
    nb_tot = 0
    for label in labels:
        bathy_label_points = bathy_points[bathy_points['depth label'] == label]
        n = 0
        nb = 0
        t = time.time()
        while nb < cfg.nb_max_pt_per_tile and n < cfg.line_max_read:
            n += 1
            n_tot += 1
            chosen_idx = np.random.choice(len(bathy_label_points))
            random_point = bathy_label_points.iloc[chosen_idx]
            if cfg.depth_lim_min <= random_point['depth(m - down positive - LAT)'] <= cfg.depth_lim_max:
                if not len(sentinel2tile_list) == 0:
                    for i in range(nb_tiles):
                        tile = sentinel2tile_list[i]
                        # proj to the good coordinate system & round to the tenth to fit on the sentinel 2 max precision
                        x_point, y_point = proj[i](random_point['long(DD)'], random_point['lat(DD)'])
                        x_point = int(round(x_point, -1))
                        y_point = int(round(y_point, -1))
                        # get the indices of the points to close to the actual point
                        ind = np.where(np.abs(np.array(x[i], copy=False) - x_point) < precision)
                        ind = np.where(np.abs(np.array(y[i], copy=False)[ind] - y_point) < precision)
                        # keep the point only if it is not to close to others
                        if len(ind[0]) == 0:
                            if isin_tile(x_point, y_point, tile.corner['x'], tile.corner['y']):
                                nb += 1
                                nb_tot += 1
                                x[i] = np.append(x[i], [x_point])
                                y[i] = np.append(y[i], [y_point])
                                z[i] = np.append(z[i], random_point['depth(m - down positive - LAT)'])
            if n % 1000 == 0:
                print(n)
        print('label :', label, nb, '/', n, time.time() - t)

    print('nb of lines read/selected :', n_tot, '/', nb_tot)

    return x, y, z
