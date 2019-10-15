import numpy as np
import os
import argparse
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from helpers import combo_npy, pre_process
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
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train for')
parser.add_argument('--dataset_indir', type=str, default="C:/Users/Brook/Desktop/Brook_tld/SIM/data/train", help='path of Raw Dataset.')
parser.add_argument('--dataset_outdir', type=str, default="C:/Users/Brook/Desktop/Brook_tld/SIM/data/train_pre", help='path of Preprocessed Dataset.')
parser.add_argument('--key', type=str, default="mask", help='key words to distinguish masks from images')
parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
parser.add_argument('--validation_splid', type=float, default=0.2, help='Percentage of dataset used for validation')
parser.add_argument('--resize_H', type=int, default=96, help='Height of cropped input image to network')
parser.add_argument('--resize_W', type=int, default=128, help='Width of cropped input image to network')
parser.add_argument('--checkpoint_dir', type=str, default="C:/Users/Brook/Desktop/Brook_tld/SIM/ckpt", help='place to store and reload checkpoints.')
parser.add_argument('--continue_training', type=str2bool, default=True, help='Whether to continue training from a checkpoint')
parser.add_argument('--model_name', type=str, default='tl_detector_model.json', help='name of saved model in the path of ckpt')
parser.add_argument('--weights_name', type=str, default='tl_detector_weights.h5', help='name of saved weights in the path of ckpt')
args = parser.parse_args(args=[])

dir_npy_imgs = args.dataset_outdir + '/tl_IMG.npy'
dir_npy_masks = args.dataset_outdir + '/tl_MASK.npy'
ckpt = args.checkpoint_dir

##################### Step 1: Data Preprocessing ##############################
print('Brook: Preparing .npy dataset. ==================')
combo_npy(args.dataset_indir, args.key, args.dataset_outdir)

print('Brook: Loading and preprocessing .npy dataset from RGB into Grey. ==================')
imgs = pre_process(dir_npy_imgs, args.resize_H, args.resize_W)
masks = pre_process(dir_npy_masks, args.resize_H, args.resize_W)

print('Brook: Optional -- Coninue to preprocess dataset by whitening ====================')
# whitening images
imgs = imgs.astype('float32')
mean = np.mean(imgs)  # mean for data centering
std = np.std(imgs)  # std for data normalization
imgs -= mean
imgs /= std
# whitening masks
masks = masks.astype('float32')
masks /= 255.  # Due to here our mask is just black or white, we can scale masks to [0, 1]

##################### Step 2: Compiling Keras model ################################
print('Brook: Creating and compiling the model. =====================')
num_channel = 1 # In my case, I has tranmsformed all RGB into greys
model = unet(args.resize_H, args.resize_W, num_channel) 

#################### Step 3: check whether there was pre-saved model ##################
#################### Step 4: Otherwise, set up the method to save model ###############
if not os.path.isdir(ckpt):
   os.makedirs(ckpt)
       
if args.continue_training:
    print('Brook: This is on continue training mode, the latest model ckpt is loading. ================')
    model.load_weights(os.path.join(ckpt, args.weights_name))
else:
    print('Brook: This is the first time training. Lets save the model. =====================')
    
model_checkpoint = ModelCheckpoint(os.path.join(ckpt, args.weights_name), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(ckpt, args.model_name), "w") as json_file:
    json_file.write(model_json)
print('Brook: Saving model to disk ========================')

#################### Step 5: Fitting Keras model ##############################
# adding a visualization function
tf_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

print('Brook: Fitting the model =========================')
model.fit(imgs,masks, batch_size=args.batch_size, epochs=args.epochs,verbose=1, shuffle=True,
              validation_split=args.validation_splid, callbacks=[model_checkpoint, tf_board])


