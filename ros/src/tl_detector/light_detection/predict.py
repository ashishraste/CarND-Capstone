import numpy as np
import os
from glob import glob
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.color import rgb2grey
import argparse
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from helpers import read_subfolder
from model import unet

##################### Step 0: Setting constant values #############################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dir_all_images', type=str, default="C:/Users/Brook/Desktop/Brook_tld/data", help='path of all Dataset.')
parser.add_argument('--dir_test_images', type=str, default="C:/Users/Brook/Desktop/Brook_tld/data/test", help='path of Testing Dataset.')
parser.add_argument('--dir_results', type=str, default="C:/Users/Brook/Desktop/Brook_tld/data/test_results", help='path of results for Testing Dataset.')
parser.add_argument('--resize_H', type=int, default=96, help='Height of cropped input image to network')
parser.add_argument('--resize_W', type=int, default=128, help='Width of cropped input image to network')
parser.add_argument('--checkpoint_dir', type=str, default="C:/Users/Brook/Desktop/Brook_tld/ckpt", help='place to store and reload checkpoints.')
parser.add_argument('--model_name', type=str, default='tl_detector_model.json', help='name of saved model in the path of ckpt')
parser.add_argument('--weights_name', type=str, default='tl_detector_weights.h5', help='name of saved weights in the path of ckpt')
args = parser.parse_args(args=[])

ckpt = args.checkpoint_dir

#################### Step 1: Reading testing data into numpy array format ########################

# Dataset structure 1: if all testing images are stored in a folder without children folders
#imgs_files = glob(args.dir_test_images + '/*.jpg')

# Dataset structure 2: if all testing images are stored in a folder with children folders
dir_test_pre = args.dir_all_images + '/test_pre'
if not os.path.exists(dir_test_pre):
    os.mkdir(dir_test_pre)
read_subfolder(args.dir_test_images, dir_test_pre)
imgs_files = glob(dir_test_pre + '/*.jpg')

eg = imread(imgs_files[0])
H = eg.shape[0]
W = eg.shape[1]
num_channels = 3
num = len(imgs_files)
imgs = np.ndarray((num, H, W, num_channels), dtype=np.uint8)

for i in range(num):
    image = imread(imgs_files[i])
    imgs[i] = np.array([image])
    
#################### Step 2: Preprocessing testing data by graying and cropping ########################
imgs_p = np.ndarray((imgs.shape[0], args.resize_H, args.resize_W), dtype=np.uint8)
for i in range(imgs.shape[0]):
    imgs_p[i] = rgb2grey(resize(imgs[i], (args.resize_H, args.resize_W, 3), preserve_range=True, mode="constant"))
imgs_p = imgs_p[..., np.newaxis]

# Optional: whitening
imgs_p = imgs_p.astype('float32')
mean = np.mean(imgs_p)  # mean for data centering
std = np.std(imgs_p)  # std for data normalization
imgs_p -= mean
imgs_p /= std

##################### Step 3: Compiling Keras model ################################
print('Brook: Creating and compiling the model. =====================')
num_channel = 1 # In my case, I has tranmsformed all RGB into greys
model = unet(args.resize_H, args.resize_W, num_channel)

#################### Step 4: Loading pre-saved model ########################
print('Brook: Loading saved weights. ==================')
model.load_weights(os.path.join(ckpt, args.weights_name))

#################### Step 5: Predicting results ########################
print('Brook: Predicting masks on test data.=============')
predicted_masks = model.predict(imgs_p, verbose=1)

#################### Step 6: Saving results ########################
print('Brook: Saving predicted masks to files. ================')
if not os.path.exists(args.dir_results):
    os.mkdir(args.dir_results)

for i in range(len(predicted_masks)):
    mask = predicted_masks[i]
    mask = (mask[:, :, 0] * 255.).astype(np.uint8)
    name = os.path.basename(imgs_files[i])
    imsave(os.path.join(args.dir_results, 'pred_'+name), mask)
    