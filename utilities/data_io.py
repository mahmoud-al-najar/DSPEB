import os
import time
import gdal
import pyproj
import tarfile
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import xml.etree.ElementTree as et
import datagen_config as cfg

from datetime import datetime, timedelta
from utilities.common import isin_tile
# from utilities.wrappers import Sentinel2Tile, Sentinel2Safe


def get_cloud_coverage(path):
    """
    Find the cloud coverage of the S2-L1C image in the xml file
    :param path:(str) path to the .SAFE repository of the image
    :return: cloud_coverage (float)
    """
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
            return tidal
        else:
            tidal = tid['prediction_at_ERA5_point'].values[0]
            return tidal


# def get_geo_transform_for_image(s2_path, tile_id, safe_id):
#     safe_path = os.path.join(s2_path, tile_id, safe_id)
#     date = safe_id[11:19]
#     t_time = safe_id[20:26]
#     a = os.listdir(os.path.join(safe_path, 'GRANULE'))
#     path = os.path.join(safe_path, 'GRANULE', a[0], 'IMG_DATA', f'T{tile_id}_{date}T{t_time}_')
#     ds = gdal.Open(path + 'B04.jp2')
#     return ds.GetGeoTransform()


def parse_sentinel2_imagesafe_metadata(safe_path):
    from utilities.wrappers import Sentinel2Safe  # TODO: UNCOMMENT MAIN IMPORT AND COME BACK TO THIS
    safe_id = safe_path.split('/')[-1]
    print(f'parse_sentinel2_imagesafe_metadata() safe_id: {safe_id}')
    date = safe_id[11:19]
    t_time = safe_id[20:26]
    tidal = get_tidal_elevation_for_image(safe_id)
    temp_safe = Sentinel2Safe()
    temp_safe.date = date
    temp_safe.time = t_time
    temp_safe.s2_path = safe_path
    x, y, epsg = get_top_left_corner_coordinates_for_image(safe_path)
    temp_safe.corners = (x, y)
    temp_safe.epsg = epsg
    if tidal:
        temp_safe.tidal_elevation = tidal
    else:
        warnings.warn(f'No tidal elevation data for safe: {safe_id}')
        temp_safe.tidal_elevation = 0
    return temp_safe


def parse_sentinel2_tiles_metadata():
    """
            This function returns the info of the nb_max_date tiles with the smallest cloud coverage
            :return: corners, paths, dates, epsgs: infos of the selected tiles
            """
    from utilities.wrappers import Sentinel2Tile  # TODO: UNCOMMENT MAIN IMPORT AND COME BACK TO THIS
    sentinel2_tiles = []
    i = -1
    for tile in cfg.tiles:
        temp_tile = Sentinel2Tile()
        temp_tile.id = tile

        print(f'--------------TILE: {temp_tile.id}')

        n = 0
        i += 1  # tile index

        path_t = os.path.join(cfg.in_path_s2, tile)
        safes = os.listdir(path_t)
        temp = []
        for safe in safes:
            if safe.endswith('SAFE'):
                temp.append(safe)
        safes = temp
        for safe in safes:
            path_s = os.path.join(path_t, safe)
            cloud_coverage = "{0:0.2f}".format(get_cloud_coverage(path_s))
            if float(cloud_coverage) < cfg.max_cc and n < cfg.nb_max_date:
                temp_safe = parse_sentinel2_imagesafe_metadata(path_s)
                if temp_safe:
                    temp_tile.safes.append(temp_safe)

                    print(f'safe.corners: {temp_safe.corners}')
                    print(f'safe.s2_path: {temp_safe.s2_path}')
                    print(f'safe.date: {temp_safe.date}')
                    print(f'safe.time: {temp_safe.time}')
                    print(f'safe.epsg: {temp_safe.epsg}')
                    print(f'safe.tidal_elevation: {temp_safe.tidal_elevation}')

                    if temp_safe.epsg not in temp_tile.epsgs:
                        temp_tile.epsgs.append(temp_safe.epsg)
                        if len(temp_tile.epsgs) > 1:
                            warnings.warn(f'==================================== Tile {temp_tile.id}: multiple epsg\'s')
                            exit()
                    if temp_tile.corner['x'] and temp_tile.corner['y']:
                        if temp_safe.corners[0] != temp_tile.corner['x'] or temp_safe.corners[1] != temp_tile.corner['y']:
                            id = temp_tile.id
                            warnings.warn(
                                f'============================================ Tile {id}: multiple corners')
                            exit()
                        else:
                            pass  # Different snapshots of the same tile should have the exact same corners
                    else:
                        temp_tile.corner['x'] = temp_safe.corners[0]
                        temp_tile.corner['y'] = temp_safe.corners[1]

                n += 1
        sentinel2_tiles.append(temp_tile)

    return sentinel2_tiles


def datagen_get_bathy_xyz(sentinel2tile_list):
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
                f'THIS SHOULD NEVER HAPPEN ====================================== didn\'t find tile {tile.id}\'s epsg?')
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
        if len(bathy_label_points) > 0:
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
                            if tile.epsgs[0] != 'EPSG:32628':
                                # proj to the good coordinate system & round to the tenth to fit on the sentinel 2 max precision
                                x_point, y_point = proj[i](random_point['long(DD)'], random_point['lat(DD)'])
                            else:
                                # bathy and s2 are already on the same coordinate system
                                x_point, y_point = random_point['long(DD)'], random_point['lat(DD)']
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
        else:
            print(f'len(bathy_label_points) at label:{label} == 0')
    print('nb of lines read/selected :', n_tot, '/', nb_tot)

    return x, y, z


def __flip_bathymetry_y_axis(arr):
    unique_values = np.unique(arr)
    flipped = np.empty(np.array(arr).shape)
    for i in range(len(arr)):
        flipped[i] = np.flipud(unique_values)[np.where(unique_values == arr[i])]
    return flipped


def read_nc_file(path_to_nc, projection_in=None, projection_out=None):
    ncd = nc.Dataset(path_to_nc)
    print(ncd)

    n_x = len(ncd.variables['x'])
    n_y = len(ncd.variables['y'])
    n_k = len(ncd.variables['kKeep'])
    n_t = len(ncd.variables['time'])

    out_x = []
    out_y = []
    out_z = []
    n_err = 0
    n_good = 0
    n_all = 0
    n_dash = 0

    for i_t in range(n_t):
        for i_x in range(n_x):
            for i_y in range(n_y):
                ncd_time = ncd.variables['time'][i_t]
                ncd_x = ncd.variables['x'][i_x]
                ncd_y = ncd.variables['y'][i_y]
                z = None
                for i_k in range(n_k):
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
                if n_all % 5000 == 0:
                    print(f'all: {n_all}, keep: {n_good}, errs: {n_err}, dash: {n_dash}')
    fn = path_to_nc.split("/")[-1]
    print(f'Filename: {fn}')
    print(f'    Total: {n_all}, 1k: {n_good}, nk: {n_err}, --: {n_dash}')
    print(f'    len(x): {len(out_x)}, len(y): {len(out_y)}, len(z): {len(out_z)}')

    print(f'    Creating CSV file for {fn}...')
    out_y = __flip_bathymetry_y_axis(out_y)
    return out_x, out_y, out_z


def read_fxyz_file(path_to_xyz, projection_in=None, projection_out=None):
    df = pd.read_csv(path_to_xyz, header=0)
    lng = np.array(df.lng)
    lat = np.array(df.lat)
    z = np.array(df.z)

    if projection_in and projection_out:
        proj = pyproj.Proj(proj='utm', init=projection_out, ellps=projection_in)
        lng, lat = proj(lng, lat)

    return lng, lat, z
