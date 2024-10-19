import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import imghdr
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

gpus = tf.config.list_physical_devices('CPU')
print(gpus) 

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus :
  tf.config.experimental.set_memory_growth(gpu,True)

# Combine List of Data
data_dir = './data'
list_dir = os.listdir(data_dir)
image_ext = ["jpeg", "jpg", "bmp", "png"]	
print(list_dir)

# Remove any dodgy images
for image_class in list_dir :
    for image in os.listdir(os.path.join(data_dir,image_class)):
       image_path = os.path.join(data_dir,image_class,image)
       try:
          img = cv2.imread(image_path)
          tip = imghdr.what(image_path)
          if tip not in image_ext:
             print("Image not in ext list {}".format(image_path))
             os.remove(image_path)
       except Exception as e:
          print("issue with the image {}".format(image_path))
        

# Load Data
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

flg,ax = plt.subplots(ncols=4,figsize = (20,20))
for idx,img in enumerate(batch[0][:4]):
   ax[idx].imshow(img.astype(int))
   ax[idx].title.set_text(batch[1][idx])