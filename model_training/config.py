region = 'saint_louis'
output_dir = 'outs'
input_csv_path = f'/media/mn/WD4TB/PEPS/data/datasets/saint_louis/{region}.csv'

seed = 1
input_shape = (80, 80, 4)
output_nodes = 1
dataset_size = None
batch_size = 64
epochs = 200
loss_function = 'mean_squared_error'

lr = 1e-05
epsilon = 1e-08
beta1 = 0.99
beta2 = 0.999
depth_normalization = 10

min_energy = 0.5
max_energy = 8
max_depth = None
