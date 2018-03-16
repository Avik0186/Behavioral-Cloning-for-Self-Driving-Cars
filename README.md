
### Exploring the Dataset

I used the dataset provided by Udacity. I could not collect more data using the training mode - it was extrmely difficult to control the vehicle in the training mode. Thus, I did not record images myself.

The dataset contains JPG images of dimensions 160x320x3. 

### Histogram of the steering angle

Histogram of the steering angle showed there were far more data of 0 steering angle thus would led to bias of driving striaght.

Thus I used a function to sample to balance_data() such that 0 steering angle has 50% of the data samples but not more. However on doing so it was observed that the vehicle moves left and right quite often than driving straight in the middle of the road. Hence balancing the data was not pursued further, it didnot lead to other undesired phenomenon. 

After the observation, I feel it makes sense since most of the data would be 0 steering angle in a normal driving environment so such bias is useful for training. 


### Data Set

I have used the Udacity provided dataset, however my model failed to take left turn just after the bridge where the right pavement is missing.
As suggested my earlier reviewer, I added recovery dataset by driving only during that portion of the track.

### My data augmentation step involved

    
   #### Step 1. Choosing either center, left or right image at random
        Using left and right camera images to simulate the effect of car wandering off to the side, and recovering. We will add a small angle offset 0.25 to the left camera and subtract a small angle offset of 0.25 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left when we are infering using only the center camera.
    
   #### Step 2. Flipping the image horizontally at random and multipling steering value with -1.0
    
    Since the dataset has a lot more images with the car turning left than right(because there are more left turns in the track), you can flip the image horizontally to simulate turing right and also reverse the corressponding steering angle.
    
   #### Step 3. Multipling the image with random brightness to augment data
   
   The brightness factor of the image is adjusted to simulate any brightness of the image to simulate driving in different lighting conditions. 
   
   #### Step 4. Adding trapezoidal random shadows as suggested in the blog 
    - https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    
    From my first submission, where the car was deviating off the road after the bridge since the right pavement is not present, to take care of such scenario used this random shadow technique. Since the shadow created would simulate the situation where the pavement is missing or the shadow due to tree or other structure should not be misleading. 
    However this augmentation did not help much in improving the model, also I could see the vehicle taking a sharp turn to the right on seeing the shadow due to the tree before the bridge.
    
   #### Step 5. Finally I preprocess the image by removing top and bottom pixels to remove irrelevant pixels and reducing noise
    
    Top 55 pixels and 25 pixels from the bottom were cropped out so that the training data contains only relevant road features. The pixels to remove was found visually through trail and error. The bottom pixels contain the hood of the car and the top portion contains the trees and other structures not important for the steering angle.

### Model Architecture -

Chosing the model was the hardest part for me in this exercise. I have explored over 25 different architectures ranging from the NVIDIA SDC model to more advanced architectures build on top of them. It deviated into the first lake on right. The following architecture worked the best for me -  

#### Layer 1: Convolution2D
#### Layer 1: relu
#### Layer 2: Convolution2D
#### Layer 2: relu
#### Layer 2: MaxPooling2D
#### Layer 3: Convolution2D
#### Layer 3: relu
#### Layer 3: Dropout = 0.4
#### Layer 4: FC 1024
#### Layer 4: Dropout = 0.3
#### Layer 4: relu
#### Layer 5: FC 512
#### Layer 5: relu
#### Layer 6: FC 1

Earlier Network architecture used - 

#### Layer 1 - convolution2d_1 (Convolution2D)
#### Layer 1 - elu_1 (ELU)
#### Layer 2 - convolution2d_2 (Convolution2D)
#### Layer 2 - elu_2 (ELU)
#### Layer 2 - dropout_1 (Dropout)
#### Layer 2 - maxpooling2d_1 (MaxPooling2D)
#### Layer 3 - convolution2d_3 (Convolution2D)
#### Layer 3 - elu_3 (ELU)
#### Layer 3 - dropout_2 (Dropout)
#### Layer 4 - convolution2d_4 (Convolution2D)
#### Layer 4 - elu_4 (ELU)
#### Layer 4 - dropout_3 (Dropout)
#### Layer 5 - flatten_1 (Flatten)
#### Layer 5 - dense_1 (Dense)
#### Layer 5 - dropout_4 (Dropout)
#### Layer 6 - dense_2 (Dense)
#### Layer 7 - dense_3 (Dense)
#### Layer 8 - dense_4 (Dense)


### Hyper-parameters and Optimization

    1. Optimizer: Adam Optimizer with learning rate of 0.0001 was found to optimal
    
    2. No. of epochs: 3. Further increase of epoch deteroriated the trained model, with 2 epoches itself the accuracy saturation of 0.055 was being attained. 
    
    3. 25,000 training images and 5000 validation images were generated using Image Augmentation process outlines earlier. 
    
    4. Keras' fit_generator method is used to train images generated by the generator so that no memory space was required by the augmented data set.

### Results and Discussion

    1. I am able to drive through track 1 with upto 17mph speed when the recording is not on and the speed is rather constant. However during recording the speed is changing quite fast.
