
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pandas as pd
from PIL import Image

LEFT_CORRECTION = 0.25
RIGHT_CORRECTION = 0.25


# In[2]:


bRunAWS = 1


# In[3]:


remove_top_pixels = 55
remove_bottom_pixels = 25
WIDTH,HEIGHT,CHANNELS = 64,64,3

# image resizing and cropping followed by normalization
def preprocess_image(collection_images):

    image = np.squeeze(collection_images[remove_top_pixels:-remove_bottom_pixels,:,:])
    image = cv2.resize(image,(WIDTH,HEIGHT))
    image = (image-128)/256
            
    return image


# In[4]:


# flip image horizontally and steering angle to simulate right turns
def flip_images(image, steering):

    image_mod = cv2.flip(image,1)
    steering_mod = steering*-1.0
    
    return (image_mod, steering_mod)


# In[5]:


# add random brightness to augment data and clip at 255
def brightness_adjust(image, steering):
    
    image_mod = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image_mod[:,:,2] = image_mod[:,:,2]*random_bright
    image_mod[:,:,2][image_mod[:,:,2]>255] = 255
    image_mod = cv2.cvtColor(image_mod,cv2.COLOR_HSV2RGB)

    return (image_mod, steering)


# In[6]:


def add_random_shadow(image):
    
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    
    if np.random.randint(2)==1:
        random_bright = .25
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    
    return image


# My data augmentation step involved
# 1. Choosing either center, left or right image at random
# 2. Flipping the image horizontally at random and multipling steering value with -1.0
# 3. Multipling the image with random brightness to augment data
# 4. Finally I preprocess the image by removing top and bottom pixels to remove irrelevant pixels and reducing noise

# In[7]:


# Data generator for reading images
def data_generator(data, batch_size):
    
    ii = 0
    N = len(data)
        
    while True:
        
        start = ii*batch_size
        end = np.amin(((ii+1)*batch_size,N))    
        data_batches_files  = data[start:end]
            
        center_image_files = np.asarray(data_batches_files['center'])
        left_image_files = np.asarray(data_batches_files['left'])
        right_image_files = np.asarray(data_batches_files['right'])
        steering_arr = np.asarray(data_batches_files['steering'])

        X_batches = np.zeros((batch_size, WIDTH, HEIGHT, CHANNELS), dtype=np.float32)
        y_batches = np.zeros((batch_size,), dtype=np.float32)
        
        for kk in range(batch_size):
        # Choose either of the image randomly
            img_type = np.random.choice(['center','left','right'])
            if(img_type=='center'):
                center_image_files[kk] = center_image_files[kk].replace('\\','/')
                file_name = center_image_files[kk].split('/')[-1]
                image = plt.imread(path+'IMG/'+file_name)
                steering = steering_arr[kk]
            elif(img_type=='left'):    
                left_image_files[kk] = left_image_files[kk].replace('\\','/')
                file_name = left_image_files[kk].split('/')[-1]
                image = plt.imread(path+'IMG/'+file_name)
                #image = plt.imread(path+left_image_files[kk][1:])
                steering = steering_arr[kk]+LEFT_CORRECTION
            elif(img_type=='right'):
                right_image_files[kk] = right_image_files[kk].replace('\\','/')
                file_name = right_image_files[kk].split('/')[-1]
                image = plt.imread(path+'IMG/'+file_name)
                #image = plt.imread(path+right_image_files[kk][1:])
                steering = steering_arr[kk]-RIGHT_CORRECTION
            
            # choose the flipped image based on random distribution   
            rand_num = np.random.random()
            if rand_num>0.5:
                image, steering = flip_images(image, steering)
            
            # add random shadows
            rand_num = np.random.random()
            if rand_num>0.5:
                image = add_random_shadow(image)    
            
            #add random brightness to image
            image, steering = brightness_adjust(image, steering)
            X_batches[kk,:,:,:] = preprocess_image(image)
            y_batches[kk] = steering
        #increment counter but reset when all images have been iterated    
        ii += 1
        if ii>=(N//batch_size):
            ii=0
    
        yield (X_batches, y_batches)


# ### Model Architecture -
# The model that worked was inspired by the NVIDIA SDC model, for some reason it didn't work for me. It deviated into the first lake on right. The subsequent modifications was able to work. 
# 
#     1. Layer 1: Convolution2D
#     2. Layer 1: relu
#     3. Layer 2: Convolution2D
#     4. Layer 2: relu
#     5. Layer 2: MaxPooling2D
#     6. Layer 3: Convolution2D
#     7. Layer 3: relu
#     8. Layer 3: Dropout = 0.4
#     9. Layer 4: FC 1024
#     10. Layer 4: Dropout = 0.3
#     11. Layer 4: relu
#     12. Layer 5: FC 512
#     13. Layer 5: relu
#     14. Layer 6: FC 1
# 
# Earlier network - 
# 
# 1. convolution2d_1 (Convolution2D)
# 2. elu_1 (ELU)
# 3. convolution2d_2 (Convolution2D)
# 4. elu_2 (ELU)
# 5. dropout_1 (Dropout)
# 6. maxpooling2d_1 (MaxPooling2D)
# 7. convolution2d_3 (Convolution2D)
# 8. elu_3 (ELU)
# 9. dropout_2 (Dropout)
# 10. convolution2d_4 (Convolution2D)
# 11. elu_4 (ELU)
# 12. dropout_3 (Dropout)
# 13. flatten_1 (Flatten)
# 14. dense_1 (Dense)
# 15. dropout_4 (Dropout)
# 16. dense_2 (Dense)
# 17. dense_3 (Dense)
# 18. dense_4 (Dense)
# 
# Total params: 8,517,473
# Trainable params: 8,517,473
# Non-trainable params: 0
# ________________________________

# In[8]:


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, ELU, Activation
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot

#nvidia SDC model as suggested in the exercise 
def nvidiaModel():
    model = Sequential()
    
    model.add(Convolution2D(24,5,5, input_shape= (HEIGHT, WIDTH, CHANNELS), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

#modified version of nvidia adding dropouts for regularization, etc.
def nvidiaModel_mod():

    model = Sequential()

    model.add(Convolution2D(24,8,8, input_shape= (HEIGHT, WIDTH, CHANNELS),subsample=(2,2)))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2)))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(48,3,3, subsample=(1,1)))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(.4))
    model.add(Dense(100))
    model.add(Dropout(.4))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

# Custom model based on trial and error and inspired from nvidia model itself
def custom_model():
    model = Sequential()

    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(16, 5, 5, input_shape= (HEIGHT, WIDTH, CHANNELS), subsample=(2, 2), border_mode="same"))
    #model.add(ELU())
    model.add(Activation('relu'))
    
    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode="valid"))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(.4))

    # layer 4 output shape is 12x12x16
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(.4))
    
    # Flatten the output
    model.add(Flatten())

    # layer 5
    model.add(Dense(1024))
    model.add(Dropout(.5))
    #model.add(ELU())
    model.add(Activation('relu'))

    # layer 6
    model.add(Dense(512))
    model.add(Dropout(.4))
    #model.add(ELU())
    model.add(Activation('relu'))
    
    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    return model

def custom_model2():
    model = Sequential()
    
    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    #model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(Activation('relu'))
    #model.add(ELU())
    #model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(Activation('relu'))
    #model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    #model.add(ELU())
    model.add(Activation('relu'))
    
    # layer 5
    model.add(Dense(512))
    #model.add(ELU())
    model.add(Activation('relu'))
    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    return model


# In[9]:


# Balance data since majority are of 0 steering angle -> done to reduce bias of the model to drive straight
def balance_data(df_log, zero_pct=0.5):
    
    total_data_size = len(df_log)
    steering_arr = np.asarray(df_log['steering'])
    zero_idx = []  
    for ii in range(total_data_size):
        if np.absolute(steering_arr[[ii]]) <= 0.25:
            zero_idx.append(ii)

    nonzero_data_size = total_data_size - len(zero_idx)
    zero_data_size = int(zero_pct * nonzero_data_size / (1 - zero_pct))

    remove_idx = np.random.choice(zero_idx, total_data_size - zero_data_size - nonzero_data_size, replace=False)
    df_log = df_log.drop(df_log.index[remove_idx]) 
    
    return df_log


# ### Training strategy
# The training strategy was to start from the NVIDIA model architecture and tweak all its hyperparameters, i.e. batch-size, data_size, epochs
# This led to straight-forward improvements of adding regularization such as drop-out, tried L2 regularization but didnot help.
# Used the intuition of gradually complex or deep but shrinking width, height kernels from earlier lecture to refine the Nvidia model. The model seems to work well however it deviates off the track after the bridge where one side of the road isnot present. To circumvent this issue tried adding the balancing data feature of dropping images with abs(steering_angle) less than 0.25. But this too didnot help. Would like to add more data/image but driving along the track, however could not even after 10-12 tries. Controlling the car in training mode is quite difficult hence had to rely completely on the images provided. 

# In[10]:


if __name__ == "__main__":

    BATCH_SIZE = 32
    
    #Select path according to where its run
    if bRunAWS==1:
        path = 'examples/data/'
    else:
        path = 'C:/Users/AVIK/Documents/Udacity Self Driving Cars/CarND-Behavioral-Cloning-P3-master/examples/data/'
    
    # read the .xls file
    df_log = pd.read_csv(path+"driving_log.csv")
    names = [name for name in df_log.columns]
    
    df_log_recovery = pd.read_csv(path+"driving_log_recovery.csv")
    names_recovery = [name for name in df_log_recovery.columns]
    df_log_recovery.rename(columns={names_recovery[0]:names[0], names_recovery[1]:names[1], names_recovery[2]:names[2],
                                names_recovery[3]:names[3],names_recovery[4]:names[4],names_recovery[5]:names[5],
                                names_recovery[6]:names[6]}, inplace=True)
    
    #drop near zero steering angle images to remove bias(added later but didn't help) 
    df_log = balance_data(df_log, zero_pct=0.80)
    df_log_recovery = balance_data(df_log_recovery, zero_pct=0.60)
    
    #df_log = df_log.append(df_log_recovery)
    
    center_image,left_image,right_image, steering = df_log['center'],df_log['left'],df_log['right'], df_log['steering']
    
    #shuffle data
    df_log = df_log.sample(frac=1).reset_index(drop=True)
    
    training_split = 0.8
    #split training and validation data after random shuffle or dataframe rows
    training_data_rows = df_log.loc[0:int(len(df_log)*training_split)]    
    validation_data_rows = df_log.loc[int(len(df_log)*training_split):]
    
    #adding the training/validation generator functions
    training_generator = data_generator(training_data_rows, batch_size=BATCH_SIZE)
    validation_data_generator = data_generator(validation_data_rows, batch_size=BATCH_SIZE)
    
    model = custom_model2() # nvidiaModel_mod()
    model.summary()

    # Compile model with Adam optimizer with learning rate as hyper-parameter. Tried - 0.0001(works best), 0.0005, 0.0008
    #model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    model.compile(loss='mse', optimizer="adam")
    
    #determines number of images/samples to iterate over during training
    samples_per_epoch = (25000//BATCH_SIZE)*BATCH_SIZE 
    
    # fit.generator to train/validate the model
    history_object = model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=5000)

    print("Saving model weights and configuration file.")

    model.save('model.h5')  
       


# ### print the keys contained in the history object
# print(history_object.history.keys())
# 
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
