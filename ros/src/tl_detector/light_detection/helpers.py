import os
from glob import glob
import cv2
from skimage.io import imsave, imread
from skimage import color
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2grey
import numpy as np

'''
The original dataset structure download from Udacity traffic bag is as below:
    ./train/
           -- green/image0.jpg, mask0.jpg, image1.jpg, mask1.jpg, ...
           -- red/image0.jpg, mask0.jpg, image1.jpg, mask1.jpg, ...
           -- yellow/image0.jpg, mask0.jpg, image1.jpg, mask1.jpg, ...
           -- none/image0.jpg, mask0.jpg, image1.jpg, mask1.jpg, ...
           
In my case, I want to do:
    1. Extract them all and split them into image and mask folder. 
    2. I will do  data augmenttaion (change brightness, add noise) onto images (except masks).
    3. Write all of images and masks into numpy.ndarry format and store as .npy files
    (remember each one image should have a corresponding mask).
    4. Loading .npy file
    5. Cropping and transforming from RGB into Gray in detestion.
'''

# Helper_Function 1: Reading images from multiple subfolders
def read_subfolder(dir_parentfolder, outdir):
    '''
    Reading all images from a parent folder with multiple chidren folders into one another specific folder
    
    @param dir_parentfolder: path to the parent folder, whose children folders contains *jpg images
    
    @param outdir: path to write the images into
    
    '''
    for subfolder in os.listdir(dir_parentfolder):
        dir_subfolder = os.path.join(dir_parentfolder, subfolder, '*.jpg')
        files = glob(dir_subfolder)
        for file in files:
            img = cv2.imread(file)
            name = os.path.basename(file)
            cv2.imwrite(os.path.join(outdir, name), img)

'''
### Example 1           
dir_parentfolder = 'C:/Users/Brook/Desktop/Brook_tld/data'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/ALL'
read_subfolder(dir_parentfolder, outdir)
###
'''

# Helper_Function 2: Split the images into different folder by name of image
def split_folder(indir, key, outdir1, outdir2):
    '''
    Spliting images and masks into different folder,
    because I want to do data augementation specifically on images later on.
    
    @param indir: path to the foler contains all images and masks
    
    @param key: the key words of name of images to be distinguished from others
    
    @param outdir1: path to folder 1, where the images will be written into
    
    @param outdir2: path to folder 2, where the masks will be written into
    
    '''
    files = glob(indir)
    for file in files:
        img = cv2.imread(file)
        name = os.path.basename(file)
        # check whether it's image or mask
        if key in name:
            cv2.imwrite(os.path.join(outdir2, name), img)
        else:
            cv2.imwrite(os.path.join(outdir1, name), img)
            
'''           
### Example 2
indir = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/ALL/*.jpg'
key = 'mask'
outdir1 = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/images'
outdir2 = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/masks'
split_folder(indir, key, outdir1, outdir2)
####
'''      
        
# Helper_Function 3: changing the brightness of images
def change_brightness(indir, outdir):
    '''
    Changing the brightness of images through hsv, and saving them into a specific folder
    
    @param indir: path to the folder contains the input images
    
    @param outdir: path to the folder write the output images into
    
    '''
    files = glob(indir)
    for file in files:
        name = 'brightness_' + os.path.basename(file)
        
        img = imread(file)
        img_hsv = color.rgb2hsv(img)
        value_layer = img_hsv[:, :, 2]
        img_hsv[:, :, 2] *= value_layer
        
        result = (color.hsv2rgb(img_hsv) * 255).astype('uint8')
        
        imsave(os.path.join(outdir, name), result)

'''       
### Example 3        
indir = 'C:/Users/Brook/Desktop/Brook_tld/data/images/*.jpg'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data/images'
change_brightness(indir, outdir)
###
'''

# Helper_Function 4: adding Gaussian noise into images
def add_noise(indir, outdir):
    '''
    Adding Gaussian noise onto images, and save them into a specific folder.
    
    @param indir: path to the folder contains the input images
    
    @param outdir: path to the folder write the output images into
    '''
    files = glob(indir)
    for file in files:
        name = 'noise_' + os.path.basename(file)
        img = cv2.imread(file)
    
        result = (gaussian(img, sigma=3, multichannel=True) * 255).astype('uint8')
        
        cv2.imwrite(os.path.join(outdir, name), result) 

'''
### Example 4        
indir = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/images/*.jpg'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data_preprocess/add_noise'
add_noise(indir, outdir)
###
'''

# Helper_Function 5: Prepare training images, including preprocessed images and masks
def prepare_data(dir_parentfolder, key, dir_masks, dir_images_tmp, dir_images):
    '''
    Reading all images and returning images and masks as output
    
    @param dir_parentfolder: path to the parent folder, whose children folders contains *jpg images
    
    @param key: the key words of name of images to be distinguished from others
    
    @param dir_masks: path to the folder to save output masks
    
    @param dir_images_tmp: path to the folder to save output images temporially
    
    @param dir_images: path to the folder to save output images
    '''
    if not os.path.isdir(dir_masks):
        os.makedirs(dir_masks)
    if not os.path.isdir(dir_images_tmp):
        os.makedirs(dir_images_tmp)
    if not os.path.isdir(dir_images):
        os.makedirs(dir_images)

    # read all images and split into 2 different folders (i.e. images and masks)
    for subfolder in os.listdir(dir_parentfolder):
        dir_subfolder = os.path.join(dir_parentfolder, subfolder, '*.jpg')
        files = glob(dir_subfolder)
        for file in files:
            img = imread(file)
            name = os.path.basename(file)
            
            # check whether it's image or mask
            if key in name:
                imsave(os.path.join(dir_masks, name), img)
            else:
                imsave(os.path.join(dir_images_tmp, name), img)
                imsave(os.path.join(dir_images, name), img)
    
    # change brightness
    change_brightness(dir_images_tmp+'/*.jpg', dir_images)      
    # add noise
    add_noise(dir_images_tmp+'/*.jpg', dir_images)
    
    # creating corresponding masks
    files = glob(dir_masks+'/*.jpg')
    for file in files:
        name1 = 'brightness_' + os.path.basename(file)
        name2 = 'noise_' + os.path.basename(file)
        img = cv2.imread(file)
        cv2.imwrite(os.path.join(dir_masks, name1), img)
        cv2.imwrite(os.path.join(dir_masks, name2), img)

'''  
### Example 5
dir_parentfolder = 'C:/Users/Brook/Desktop/Brook_tld/data'
key = 'mask'
dir_masks = 'C:/Users/Brook/Desktop/Brook_tld/data/masks'
dir_images_tmp = 'C:/Users/Brook/Desktop/Brook_tld/data/images_tmp'
dir_images = 'C:/Users/Brook/Desktop/Brook_tld/data/images'
prepare_data(dir_parentfolder, key, dir_masks, dir_images_tmp, dir_images)
###   
'''   

# Helper_Function 6 Prepare .npy files  
def prepare_npy(dir_images, dir_masks, outdir):
    '''
    prepare .npy files for images (including augmented ones) and masks
    
    @param dir_images: path to images
    
    @param dir_masks: path to masks
    
    @param outdir: path to save the output .npy files
    '''
    
    imgs_files = glob(dir_images+'/*.jpg')
    masks_files = glob(dir_masks+'/*.jpg')
    
    # Initializing imgs and masks
    eg = cv2.imread(masks_files[0])
    H = eg.shape[0]
    W = eg.shape[1]
    num_channels = 3
    num = len(masks_files)
    imgs = np.ndarray((num, H, W, num_channels), dtype=np.uint8)
    masks = np.ndarray((num, H, W, num_channels), dtype=np.uint8)
    
    print ('Brook: I am creating .npy dataset ==================')
    for i in range(num):
        image = imread(imgs_files[i])
        mask = imread(masks_files[i])
        imgs[i] = np.array([image])
        masks[i] = np.array([mask])
        
    print('Brook: Loading done. =================')
    np.save(os.path.join(outdir, 'tl_IMG.npy'), imgs)
    np.save(os.path.join(outdir, 'tl_MASK.npy'), masks)
    print('Brook: Saving to .npy files done. ======================')

'''
### Example 6
dir_masks = 'C:/Users/Brook/Desktop/Brook_tld/data/masks'
dir_images = 'C:/Users/Brook/Desktop/Brook_tld/data/images'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data'
prepare_npy(dir_images, dir_masks, outdir)
###   
'''

# Helper_Function 7: prepare .npy file with a combo function
def combo_npy(indir, key, outdir):
    '''
    prepare .npy files for images (including augmented ones) and masks
    
    @param indir: path to the parent folder, whose children folders contains *jpg images
    
    @param key: the key words of name of images to be distinguished from others
    
    @param outdir: path to save the output .npy files
    
    '''
    dir_masks = outdir + '/masks'
    dir_images_tmp = outdir + '/images_tmp'
    dir_images = outdir + '/images'
    
    prepare_data(indir, key, dir_masks, dir_images_tmp, dir_images)
    prepare_npy(dir_images, dir_masks, outdir)

'''   
### Example 7
indir = 'C:/Users/Brook/Desktop/Brook_tld/data'
key = 'mask'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data_pre'

combo_npy(indir, key, outdir)
###
'''

# Helper_Function 8: load the .npy file
def load_npy(dir_npy):
    '''
    Loading the .npy file back into array.
    
    @param dir_npy: path to folder contains .npy files
    '''
    imgs = np.load(dir_npy)
    return imgs

'''
### Example 8
dir_npy_imgs = 'C:/Users/Brook/Desktop/Brook_tld/data/train_pre/tl_IMG.npy'
imgs = load_npy(dir_npy_imgs)
dir_npy_masks = 'C:/Users/Brook/Desktop/Brook_tld/data/train_pre/tl_MASK.npy'
masks = load_npy(dir_npy_masks)
###
'''

# Helper_Function 9: Cropping and Transforming RGB into Gray after loading .npy file
def pre_process(dir_npy, resize_H, resize_W):
    '''
    @param dir_npy: path to .npy file
    
    @param resize_H: resize height of original image into resize_H
        
    @param resize_W: resize width of original image into resize_W
    '''
    imgs = load_npy(dir_npy)
    
    imgs_p = np.ndarray((imgs.shape[0], resize_H, resize_W), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = rgb2grey(resize(imgs[i], (resize_H, resize_W, 3), preserve_range=True, mode="constant"))

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

'''
### Exmaple 9
dir_npy_imgs = 'C:/Users/Brook/Desktop/Brook_tld/data_pre/tl_IMG.npy'
dir_npy_masks = 'C:/Users/Brook/Desktop/Brook_tld/data_pre/tl_MASK.npy'
resize_H, resize_W = 96, 128
imgs_p = pre_process(dir_npy_imgs, resize_H, resize_W)
masks_p = pre_process(dir_npy_masks, resize_H, resize_W)
'''

### Conclusion
'''
In this project of traffic light detection, we can easily applied the following functions to complete data preprocessing:
    
    1. combo_npy(indir, key, outdir)
    
    2. pre_process(dir_npy, resize_H, resize_W)
    
For example,

indir = 'C:/Users/Brook/Desktop/Brook_tld/data'
key = 'mask'
outdir = 'C:/Users/Brook/Desktop/Brook_tld/data_pre'

combo_npy(indir, key, outdir)

resize_H, resize_W = 96, 128
dir_npy_imgs = outdir + '/tl_IMG.npy'
dir_npy_masks = outdir + '/tl_MASK.npy'

imgs_p = pre_process(dir_npy_imgs, resize_H, resize_W)
masks_p = pre_process(dir_npy_masks, resize_H, resize_W)

'''


