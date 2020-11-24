import numpy as np
import keras
import csv
import model_training.config as cfg
import math
import random
np.random.seed(448)


class ExtractsGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_ids, labels, batch_size=1, x_shape=cfg.input_shape, y_size=cfg.output_nodes, shuffle=True):
        """Initialization"""
        self.x_shape = x_shape
        self.y_size = y_size
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indices = None
        self.on_epoch_end()
        self.shape = self.x_shape

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        y = np.empty((self.batch_size, self.y_size))
        # Generate data
        for i, ID in enumerate(list_ids_temp):
            burst = np.load(ID)
            x[i, ] = burst
            y[i] = float(self.labels[ID]) / cfg.depth_normalization
        return x, y


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


# TODO: continue
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
