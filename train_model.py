import model_training.config as cfg
from model_training.extracts_generator import ExtractsGenerator
from model_training.read_data import get_data_partitions, get_ids_list_and_labels_dict
from model_training.resnet import ResNet
import numpy as np
import random
import os

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = cfg.lr

    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


random.seed(cfg.seed)
np.random.seed(cfg.seed)

train_ids, train_labels = get_ids_list_and_labels_dict(cfg.path_csv_train)
val_ids, val_labels = get_ids_list_and_labels_dict(cfg.path_csv_validation)
test_ids, test_labels = get_ids_list_and_labels_dict(cfg.path_csv_test)

print('Data partitions:')
print('    Training: ' + str(len(train_ids)))
print('    Validation: ' + str(len(val_ids)))
print('    Test: ' + str(len(test_ids)))

generator_train = ExtractsGenerator(train_ids, train_labels)
generator_validation = ExtractsGenerator(val_ids, val_labels)
generator_test = ExtractsGenerator(test_ids, test_labels)

total_items = len(train_ids)
num_batches = int(total_items / cfg.batch_size)

resnet = ResNet(input_shape=cfg.input_shape, output_nodes=cfg.output_nodes, n=6)
model = resnet.create_model()

# print(cfg.region)
# model.summary()

output_dir = model.name + '_' + str(total_items) + '_lr_' + str(cfg.lr) + '__epsilon_' + \
             str(cfg.epsilon) + '__beta1_' + str(cfg.beta1) + '__beta2_' + str(cfg.beta2) + '/'
output_dir = os.path.join(cfg.output_dir, output_dir)
if output_dir != '':
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

# Prepare callbacks for model saving and for learning rate adjustment.
file_path = output_dir + model.name + '_' + str(total_items) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
csv_logger = CSVLogger(output_dir + 'training.log', separator=',')
lr_scheduler = LearningRateScheduler(lr_schedule)
tb_cb = TensorBoard(log_dir=output_dir, histogram_freq=0, write_graph=True, write_images=True,
                    update_freq='epoch')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

callbacks = [csv_logger, checkpoint, lr_scheduler, es]  # , tb_cb, lr_reducer, lr_scheduler]

# Compile and train model
model.compile(loss=cfg.loss_function,
              optimizer=Adam(lr=lr_schedule(0), epsilon=cfg.epsilon, beta_1=cfg.beta1, beta_2=cfg.beta2))

history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches,
                              epochs=cfg.epochs, verbose=1, validation_data=generator_validation,
                              callbacks=callbacks)

scores = model.evaluate_generator(generator=generator_test)
print(scores)
