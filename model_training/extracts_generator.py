import keras
import numpy as np
import model_training.config as cfg
from data_generation.preprocessing import apply_per_band_min_max_normalization
np.random.seed(448)


class ExtractsGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_ids, labels, batch_size=cfg.batch_size,
                 x_shape=cfg.input_shape, y_size=cfg.output_nodes, shuffle=True):
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
            if not ID.endswith('.npy'):
                burst = np.load(f'{ID}.npy')
            else:
                burst = np.load(ID)

            if cfg.input_shape[-1] == 2:
                _temp_burst = np.empty((self.x_shape[0], self.x_shape[1], 2))
                _temp_burst[:, :, 0] = burst[:, :, 0]
                _temp_burst[:, :, 1] = burst[:, :, 3]
                burst = _temp_burst

            if cfg.normalize_input:
                # Normalize
                burst = apply_per_band_min_max_normalization(burst)

            # Rotate
            burst = np.rot90(burst, k=np.random.choice(4))

            x[i, ] = burst
            y[i] = float(self.labels[ID]) / cfg.depth_normalization
        return x, y
