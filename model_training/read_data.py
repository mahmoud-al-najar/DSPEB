import csv
import math
import random
import numpy as np
import model_training.config as cfg

random.seed(cfg.seed)
np.random.seed(cfg.seed)


def get_ids_list_and_labels_dict():
    with open(cfg.input_csv_path, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    random.shuffle(dataset)

    list_ids = []
    dict_labels = dict()

    for r in dataset:
        ## 0: path, 1: z, 2: x, 3: y, 4: epsg, 5: max_energy, 6: z_no_tide
        if r[1] == 'z':
            continue
        elif cfg.max_depth and float(r[1]) > cfg.max_depth:
            continue
        elif cfg.min_energy and float(r[5]) < cfg.min_energy:
            continue
        elif cfg.max_energy and float(r[5]) > cfg.max_energy:
            continue
        else:
            list_ids.append(r[0])
            dict_labels[r[0]] = float(r[1])
            if cfg.dataset_size and len(list_ids) > cfg.dataset_size:
                break
    return list_ids, dict_labels


def get_data_partitions(list_ids):
    # Setup data generators
    train_test_ratio = 0.9
    validation_split = 0.2
    train_size = math.floor(train_test_ratio * len(list_ids))
    validation_size = train_size * 0.2

    partition = dict()
    partition['train'] = list_ids[:int(train_size - validation_size)]
    partition['validation'] = list_ids[int(train_size - validation_size):int(train_size)]
    partition['test'] = list_ids[int(train_size):]
    return partition
