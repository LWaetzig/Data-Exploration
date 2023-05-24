import os

img_width = 30
img_height = 30
n_channels = 3
batch_size = 32
n_epochs = 200
val_batch_size = 32
class_names = list(range(43))

train_path = os.path.join('data', 'Train')
test_path = os.path.join('data', 'Test')

all_image_paths_train = list(train_path.glob('*/*'))
all_image_paths_train = [str(path) for path in all_image_paths_train]