import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.utils.np_utils import to_categorical
from keras import losses, optimizers, regularizers
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


###################################
### Step 0: Setting constant values 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='Number of images in each batch')
parser.add_argument('--dir_data', type=str, default='C:/Users/Brook/Desktop/Brook_tlc/SIM/data/train', help='path of Raw Dataset.')
parser.add_argument('--validation_split', type=float, default=0.1, help='Percentage of dataset used for validation')
parser.add_argument('--resize_H', type=int, default=64, help='Height of cropped input image to network')
parser.add_argument('--resize_W', type=int, default=32, help='Width of cropped input image to network')
parser.add_argument('--checkpoint_dir', type=str, default="C:/Users/Brook/Desktop/Brook_tlc/SIM/ckpt", help='place to store and reload checkpoints.')
parser.add_argument('--weights_name', type=str, default='tl_classifier_weights.h5', help='name of saved weights in the path of ckpt')
parser.add_argument('--mode', type=str, default='predict', help='Train or predict')
parser.add_argument('--continue_training', type=str2bool, default=None, help='Whether to continue training from a checkpoint')
parser.add_argument('--dir_test_imgs', type=str, default='C:/Users/Brook/Desktop/Brook_tlc/SIM/data/test_imgs', help='path of testing Dataset.')
parser.add_argument('--dir_test_results', type=str, default='C:/Users/Brook/Desktop/Brook_tlc/SIM/data/test_results', help='path of reults of testing Dataset.')
args = parser.parse_args(args=[])

dir_sim_red = os.path.join(args.dir_data,'sim_red')
dir_sim_yellow = os.path.join(args.dir_data,'sim_yellow')
dir_sim_green = os.path.join(args.dir_data,'sim_green')
dir_sim_none = os.path.join(args.dir_data,'sim_none')
H = args.resize_H
W = args.resize_W
ckpt = args.checkpoint_dir

label_dict = ['Red', 'Yellow', 'Green', 'None']
num_classes = len(label_dict)

#######################
### Step 1 Loading data and preprocessing data

X_train = []
x_label = []
for img_class, directory in enumerate([dir_sim_red, dir_sim_yellow, dir_sim_green, dir_sim_none]):
    for i, file_name in enumerate(glob.glob("{}/*.jpg".format(directory))):
        file = cv2.imread(file_name)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(file, (W,H))   # resize

        X_train.append(resized/255.)    # normalization
        x_label.append(img_class)
        
X_train = np.array(X_train) # numpy arraying
x_label = np.array(x_label)

#######################       
### Step 2 Building CNN (writing code in format of Keras sequential model)

def BrookNN():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(H, W, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2,2))
    Dropout(0.8)
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2,2))
    Dropout(0.8)
    model.add(Flatten())
    model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, activation='softmax'))
    loss = losses.categorical_crossentropy
    optimizer = optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model

model = BrookNN() # Keras model: The first step is always to compile model, then go for either train model or load saved model

#######################       
### Step 3 Training
    
if args.mode == 'train':
    
    if not os.path.exists(ckpt):
        os.mkdir(ckpt)
        
    # Check whether there was pre-saved model
    if args.continue_training:
        print('Brook: This is on continue training mode, the latest model ckpt is loading.')
        model.load_weights(os.path.join(ckpt, args.weights_name))
    else:
        print('Brook: This is the first time training. Lets train and save the model.')
    
    # train the model
    categorical_labels = to_categorical(x_label)  # one-hot label (https://keras.io/utils/#to_categorical)
    model.fit(X_train, categorical_labels, batch_size=args.batch_size, epochs=args.epochs, verbose=True, validation_split=args.validation_split, shuffle=True)

    # Optional: evaluate the model
    score = model.evaluate(X_train, categorical_labels, verbose=True)
    print(score)

    # save the model
    model.save(os.path.join(ckpt, args.weights_name))

#######################       
### Step 4 predict

elif args.mode == 'predict':
    print('Brook: This is on the mode of prediction. Lets load previously trained model')
    model.load_weights(os.path.join(ckpt, args.weights_name))
    
    # loading testing imgs and cropping them and writing them into 4D array format
    # suppose all testing images are saved in one specific folder
    Y_test = []
    files = glob.glob(args.dir_test_imgs + '/*.jpg')
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        resized = cv2.resize(img, (W,H))  # resize
        Y_test.append(resized/255.)        # normalization
    
    Y_test = np.array(Y_test)
    
    # prediction
    prediction = model.predict(Y_test, verbose=1)
    
    # Optional: saving the reults into a list in a .txt file 
    print('Brook: Saving predicted results to files.')
    if not os.path.exists(args.dir_test_results):
        os.mkdir(args.dir_test_results)
    txt_file = open(os.path.join(args.dir_test_results, 'predict.txt'), "w") 
            
    # displaying predicted results numerically
    for i in range(len(prediction)):
        print ('Index: ', i, 'Predicted results: ', prediction[i])
        plt.imshow(Y_test[i])
        
        img_name = files[i]
        max_index = list(prediction[i]).index(max(prediction[i]))
        string = img_name + ':  ' + label_dict[max_index]
        txt_file.write(string)
        txt_file.write('\n')
    txt_file.close()
    
    

    


    

