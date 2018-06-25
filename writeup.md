# **Traffic Sign Recognition**

## Writeup



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./3traffic_sign_distribution.png "Visualization 1"
[image2]: ./3traffic_sign_examples.png "visualization 2"
[image3]: ./grayscale.png "Grayscaling"
[image4]: ./data_augmentation.png
[image5]: ./cropped.png "Traffic Sign 1"
[image6]: ./network_architecture.png "Traffic Sign 3"
[image7]: ./internet_images.png "Traffic Sign 4"
[image8]: ./top5pred.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1.  The Data Set

### Data Set Summary & Exploration

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. A bar chart showing the number of samples per classes indicates a slight skew:

![alt text][image1]

We also plot the first image of each class in the training set:

![alt text][image2]

This gives us an idea of what kind of pictures we're dealing with. Clearly the lighting and resolution is not great in some of them.

### 2. Preprocessing 

To begin with our NumPy array of training images, X_train, has shape (34799,32,32,3). 

As a first step, I decided to convert the images to grayscale because with the adverse lighting conditions, some of them already appear very dark. Therefore training on colours seems a little redundant; if it is looking for a red sign in order to make a particular classification, this is not guaranteed in the test set and so the model could get confused. It seems wiser to train it to recognise shapes rather than colours. The images are now (32,32,1)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

The NumPy array X_train now has shape (34799,32,32,1).

Next I normalized the image data. After grayscaling, the pixel data is in the range [0,255]. Normalising brings this to a range of [-1,1]. This way, each feature value essentially corresponds to a z-score. The reason for doing this is that during training, we will be multiplying these features by weight matrices and then adding biases (forward propagation) before backpropagating the resulting errors through the network to update the gradients. If our features aren't on a similar scale with one another, the gradients may go out of control. One way to control this would be to have different learning rates for different gradients but it is much simpler to sacle the features and then just use a single learning rate. An alternate way of thinking about this is that neural nets share many parameters and if the pixel data wasn't scaled this way then our weight matrices might have massive effects on one part of the image and miniscule effects on other parts.

As mentioned above, the data is somewhat skewed with some classes seriously under-represented in comparison to others. This could lead to problems since the network may pay less attention to learning the features of the classes with lower frequencies. To augment the dataset, I applied a small but random amount of transformation, scaling, warping (affine transformation i.e. matrix multiplication) and brightening to images from classes with less than 800 samples This was repeated until there were exactly 800 samples in the class. It is important to make random transformations so that the created images are genuinely "new" and different from the other newly  created images. At the same time, we must be careful to make only small changes to avoid producing augmented data that is unrecognisable - this would dangerously alter the classifier's interpretation of what signs from those classes look like. Below we show what each of the augmentation steps does individually, and then in the final picture we show their combined effect which is what we actually use. Notice how subtle these transformations are; apart from rotations or translations of the border, they are virtually unnoticeable.

![alt text][image4]

The NumPy array X_train now has shape (46480,32,32,1).

Finally, I decided to crop the images to (26,26,1) by removing a 3 pixel margin around each side. This is relatively simple to perform on the entire 4-dimensional NumPy array X_train. We must remember to also do this to X_valid and X_test as these will be fed into the same TenssorFlow graph.

Here is an example of an original image and a cropped image:

![alt text][image5]

The NumPy array X_train now has shape (46480,26,26,1).


#### 3. Convolutional Neural Network Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
|							|
| Preprocessing     	| 26x26x1 grayscale image                       |                                               |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 26x26x6 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 26x26x16    |
| ReLU    	|                        |                                               |
| Inception Modulue     	|  outputs 26x26x256                       |                                               |
| Max pooling	      	| 2x2 stride,  same padding, outputs 13x13x256 				|
| Convolution 1x1	    | 1x1 stride, same padding, outputs 13x13x256      									|
| Flatten		| length  43264                             |                                               |
| Fully connected		| length 512        									|
| Softmax(Logits)			| length 43       									|
|					|												|
|						|												|
This can be visualised as:

![alt text][image6]

The inception module itself took the output of the preceeding 3x3 convolution layer and applied a 5x5 convolution, 3x3 convolution, 1x1 convolution and 3x3 maxpooling followed by 1x1 convolution simultaneously. Each of these convolutions applied 64 filters. The results were concatenated together to give a 26x26x256 output that was then passed to the next network layer (2x2 maxpool).

####  4. Training

The largest batch size that was compatible with the g2.2xlarge AWS instance was 512. I used an Adam Optimizer and trained the network for 20 epochs. I used an initial learning rate of 0.0009 but reduced it by 80% after 5 epochs, then by 80% again after 10 epochs, and by a further 60% after 15 epochs. This adaptive learning rate makes the network less likely to overshoot the minimum of the loss function.

#### 5. Optimizing the Network

My final model results were:
* validation set accuracy of 96.9%
* test set accuracy of 95.5%

Already having some experience with CNNs, I wanted to try and implement an Inception Module immediately as part of my CNN architecture. Initially I made a CNN that involved two separate Inception Modules as well as multiple convolution and pooling layers. This architecture kept causing memory overflow on the g2.2xlarge AWS instance, even with a tiny batch size.

I then gave up and tried a traditional CNN with a pyramid of convolution and pooling layers. This trained extremely fast and obtained an test set accuracy of ~90%.

To improve, I tried again to implement a CNN with an Inception Module, but this time using only one of them as well as a smaller number of supporting convolution and pooling layers than before. This worked and was able to support a healthy batch size of 512.

It was then just a case of playing around with the number of epochs; trying to set it sufficiently high that the model can learn as much as possible but not too high that it starts to overfit (seen by decreasing validation accuracy). At this point, I played around with my adaptive learning rate, trying out different points at which I would reduce it. Once I hit a test set accuracy of >95%, I was happy enough.

### 6. Testing the Model on New Images

#### Internet Images of German traffic signs and a qualitative discussion of what features might be easy/difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image7]

I would expect the model to be good at recognising the "Priority Road" sign (5th image when reading left to right) as this has a unique shape.

I would expect that it might find speed limits the most challenging to tell apart since it is fairly easy to recognise the circle with a border but then it has to perform an "MNIST-y" digit classification on top of that. The 60kph sign with snow on top will be particularly difficult.

Similarly, I would expect the model finds it easy to identify a triangular sign but then to distinguish between them it must be able tor ecognise the central black image. However, I would argue that this is easier than the speed limit signs since the differences are more pronounced.

The turning signs should also be fairly easy to tell apart.  They have a circular shape with a dark background which should be easy to identify. Then to tell them apart from each other, it is a matter of recognising the arrow direction; again this is not as subtle as the digits in the speed limits.


#### Model's predictions on these new traffic signs 

The model is able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.9%. This compares favourably to the accuracy on the test set of 95.5%. The one that it got wrong was difficult due to weather conditions (more detail provided in next section).


####  Model confidence on new traffic signs

Here are the top 5 softmax predictions for each of the internet images:

![alt text][image8]


In more depth:


* Model is 100% confident about its (correct) predictions for 5/9 images (images 1,2,4,5,9).
* For image 3, it is 99% confident and its second guess is <0.5%.
    For image 6 it predicts correctly but with lower confidence. It's other guesses are also triangular signs with borders so it appears the part that is causing it some hesitation is the black image in the centre. I'd reiterate that it did predict correctly, but perhaps by tweaking the model some more, it could be more confident about it.
*   For image 7, it also predicts correctly but with lower confidence. This might be expected since the speed limit signs are all very similar so it must first identify it's a speed limit sign and then perform an "MNIST-y" type classification of the digits inside. We see this with image number 3 also.
    The interesting thing about image 7 is that it is 35% confident this "30kph" sign is in fact a "STOP" sign. Again, I reiterate that it did predict correctly here just with less certainty. The good news is that if the car were to make this confusion, it's not too dangerous; it might block the flow of traffic by stopping in a 30kph zone (generally an urban area) but at least it's not confused it with a 60kph zone. Again, with further training, this confidence could probably be improved.
*   Lastly there is image 8 which the model makes an incorrect prediction for. This is by far the most difficult image since the sign is covered in snow and that reduces the visibility substantially. Indeed, this is often a task that humans struggle with. We see here that the correct class ("60kph") doesn't even make it into the top three predictions.
    However, it is 65% sure that it's a speed limit sign (a "20kph" sign to be precise). Again you might think this is not too dangerous since driving slow makes sense in the snow! Jokes aside, this does highlight the difficulty that a self-driving car could face in adverse weather conditions. Of course, I'm sure that car manufacturers are well aware of this and one solution would be to train a model using "winter images" to address this.

### 