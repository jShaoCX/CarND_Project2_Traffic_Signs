#**Traffic Sign Recognition** 
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

[image0]: ./project_images/explore_image.png "explore_image_sign"

[image1]: ./project_images/beware_ice.png "beware_ice_sign"
[image2]: ./project_images/no_entry.png "no_entry_sign"
[image3]: ./project_images/pedestrians.png "pedestrians_sign"
[image4]: ./project_images/road_work.png "road_work_sign"
[image5]: ./project_images/speed_30.png "speed_30_sign"

[image6]: ./project_images/training_data_graph.png "training_data_graph"
[image7]: ./project_images/validation_data_graph.png "validation_data_graph"
[image8]: ./project_images/test_data_graph.png "test_data_graph"

[image9]: ./project_images/augment_brightness.png "augment_brightness_test"
[image10]: ./project_images/augment_resize.png "augment_resize_test"
[image11]: ./project_images/augment_rotate.png "augment_rotate_test"
[image12]: ./project_images/augment_shear.png "augment_shear_test"
[image13]: ./project_images/augment_translate.png "augment_translate_test"

[image14]: ./project_images/histogram_equalization_image.png "histogram_equalization_image_test"

[image15]: ./project_images/TrialData_image.png "Trial_Data"
[image16]: ./project_images/classification_report.png "classification_report"

[image17]: ./project_images/softmax_beware_ice_snow.png "softmax_beware_ice_sign"
[image18]: ./project_images/softmax_no_entry.png "softmax_no_entry_sign"
[image19]: ./project_images/softmax_pedestrians.png "softmax_pedestrians_sign"
[image20]: ./project_images/softmax_road_work.png "softmax_road_work_sign"
[image21]: ./project_images/softmax_speed_limit.png "softmax_speed_30_sign"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jShaoCX/CarND_Project2_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. The following are the histograms of the data distribution of the test set, validation set and the training set. There is a relatively uneven distribution of data across the types of signs but that can be remedied with data augmentation. However, the distribution of the three sets (training, validation and test) are similar so it could be possible to keep the distributions the same across the three sets to improve accuracy, though that would not be a good way to go since future test sets could come in different distributions.

![alt text][image6]
![alt text][image7]
![alt text][image8]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

I attempted to integrate image normalization into my pipeline but normalization actually decreased my accuracy in conjunction with my data augmentation. I looked online and some sites explained the absolute need for image normalization and 0-centering just like all other feature scaling. It would allow for the neural network to converge faster by providing a uniform gradient across all dimensions. Otherwise, the optimizer could potentially oscillate across one particular dimension while progressing very little towards the global minima. However, for this particular case, the images are from 0 to 255 and some other sites did not place a large emphasis on normalization because the data range is relatively small. I found that with a plain Lenet and image normalization, the network's validation accuracy increased by about 1.5%, however, I repeatedly attempted to use it in conjunction with my data augmentation with opencv's warp affine function and I could not replicate the same success. I have the image normalization strategy and use inside my python notebook commented out. I could not determine what the issue was.   

I did not integrate grayscaling into my pre-processing because I felt that the RGB channels of data would provide useful information about the sign. Averaging out the color channels into a single channel did not provide significant improvement to my neural network, though it did make normalization of images more straightforward.

I also tried to use histogram equalization by converting the image to HSV colorspace and then performing histogram equalization on the V channel and it helped my network converge quicker and added a slight improvement on the validation accuracy (about 0.5%). Making the image clearer by improving contrast could make it easier for the filters to capture vital edges in the small pictures in the middle of signs. But with a 32x32 image, those small insignias were extremely pixelated and could not become more detailed even with histogram euqalization. This technique may be more suited for larger images and shapes. In the end, I decided to keep the histogram equalization as it was one of the few preprocessing steps that did not have any adverse effects on my accuracy. An example of a histogram equalization image is show below:

![alt text][image14]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The splits between training, valiation and test data were already pre-determined by the pickles given in the project. Though if I had to split the data, I would use either the sklean split function and probably dedicate 60% to training, 20% to validation and 20% to testing. 

The sixth, seventh and eight code cells of the IPython notebook contain the code for augmenting the data set. The sixth cell is mainly just testing what range of augmentation is realistic. The augmentations include translation, rotation, shearing, resizing and brightness augmentation. The seventh code cell was meant for color manipulation and includes the brightness augmentation. Histogram equalization was added as an augmentation but became a mandatory preprocess step. The remnants of a data augmentation set are still present in the code. I attempted to generate a much larger data set by augmenting the existing data and simply appending it to the original data set but that quickly exploded in memory. Not only that, it was prone to overfitting once the network became deep enough because the data set, though large, was still static in size. This led me to create cell eight which is a data augmentation generator. This generator has a 1/4 probability of not augmenting the data at all and the other portion of the time, it would apply random transformations and color augmentation. I found this to be most effective to keep a small portion of the original data. 100% data augmentation with my current network's depth and size was prone to never being able to achieve high training accuracy. This seems to be reasonable since if the images look different every time, the network could have a hard time determining which features of the image are important. I also attempted to do data balancing with the data augmentation by including a Counter object for the labels. This Counter would track the average number of samples per label type and augmenting only data under the average and then incrementing the Counter by one for that particular low frequency label. Eventually the standard deviation of the data would decrease to below one and the generator would just augment all incoming data after that. This did not prove very useful because the distribution of the test data was about the same as the training data. More importantly, the network has a very low chance of training on the original training images which contains vital information. The performance was higher for just random chance augmentation. 

Here is an example of an original image and an augmented image:

![alt text][image0]

The difference between the original data set and the augmented data set is the following ... 

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x18   |
| ReLU					|												|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 6x6x27     |
| ReLU					|												|
| Max pooling			| 2x2 stride,  outputs 3x3x27       			|
| Fully connected		| outputs 243									|
| ReLU					|												|
| Fully connected		| outputs 243									|
| ReLU					|												|
| Fully connected		| outputs 43									|
| Softmax				| (part of the loss operation cell)				|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth and 11th cell of the ipython notebook. 

Because this is a network for classification, the softmax and cross entropy from tensorflow was used. From papers found online and cs231n, the ADAM optimizer seems to be the best one for now so I just used the same structure from the lenet python notebook. I used a learning rate of 0.001 since the network seemed to converge quite quickly already (around 10th to 12th epoch). Raising it to 0.005 and 0.01 caused the network to decrease in accuracy because as stated before, the network was already able to converge very quickly with 0.001. Raising the learning rate probably caused the network to oscillate around the global minima and never be able to converge towards it. Decreasing the learning rate, did not help significantly since the network did not converge at a point lower than 99.1% - 99.8% training accuracy. I believe this means that the network is reaching the global optima and any lower learning rate would just waste time. The batch size, I raised to 512 just to get a larger sample of data to have a better chance of stepping towards the global minima each time. Too small of a batch size and the calculated step may not take into account all of the dimensions of the data and make a step towards a local minima instead of the global minima. My computer was able to handle 512 without slowing down too much and raising it to 1024 did significantly slow down my network's speed. I used 35 epochs just to see if the model would converge further but as the progression in the python notebook reveals, 25 to 30 epochs is enough.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 12th and 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 95.3% 
* test set accuracy of 94.4%

I took an iterative approach which resulted in two avenues of trials, one larger neural network and the other based on Lenet. The experiments ended up ultimately converging back into a single network with a few elements of both experimental branches. The trial graph is show below and is contained in the TrialData excel file in the folder.

![alt text][image15]

* What was the first architecture that was tried and why was it chosen? What were some problems with the initial architecture?
I first experimented with the default lenet architecture and image normalization but found that it converged at around 86% test accuracy without any other modifications. However, the training accuracy reached 97% so I figured that some dropout would help. With 0.7 keep probability the network actually performed worse so other adjustments had to be made. Furthermore, when I added data augmentation, the simple lenet was unable to fit the augmented data. I believe the smaller network was underfitting the large variation of data coming in through the augmentation generator.
*  How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I felt that the original architecture was too small and was underfitting the data so I split off an experimental file with a slightly larger network and smaller convolutions. This is shown in my TrialData image above. With a larger network and some data augmentation (just increasing the size of the data set), the accuracy got worse. Dropout turned to be the main cause of the low accuracy. It is possible that with a large portion of the data augmented, the network was unable to train properly with even a small amount of dropout. I found that adjusting various parts of the network such as convolution size and number of filters was largely ineffective at the range of 6-27. I assume that if the network was much much larger as I've seen online (up to 128 filters for some layers), the network may have behaved differently. So fine tuning these small parameters (trials 4-8) was not very effective. I also discovered the issue that I mentioned before, where normalization decreased the accuracy of my network. Again, I was not able to get to the bottom of it but I assume it may be the way openCV interacts with normalized data. It is only in conjunction with my data augmentation that the issue arose. I was careful to only normalize after the data augmentation occured and augment all data (train, test and validation) in the same way. I printed some of the actual pixel values of the normalization and display the normalized images as well but could not find anything abnormal. Eventually, without dropout and more consistent data augmentation, I was able to get the larger neural network to converge at 100% training accuracy and validation of 91.3% validation accuracy. I was stuck in this range for quite a while where it overfit the training data but was unable to make progress on new data. I also experimented with Xavier initialization but the differences between that and a truncated normal initialization with small standard deviation was insignificant. 
* Which parameters were tuned? How were they adjusted and why?
As stated above, the parameters such as valid or same padding, 3x3 or 5x5 convolutions, an extra convolution layer, or 6-32 filters per layer were fairly insignificant after the network became at least twice as big as the lenet. The original lenet was definitely too small to fit the augmented data though it was sufficient to overfit to just the plain training data (34000 or so samples). I tried varying the fully connected input and output neuron counts but that did not prove very useful either at the range of 250-2000. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think the most important design choice for me at the last few trials was taking the original lenet and adjusting it incrementally. I kept the original layers and just added a few more filters. Then after I found the network would overfit the augmented data, I added another convolution layer after the 2nd convolution layer to make the network slightly deeper. This proved to be the most significant improvement, putting me in the passable mid 90% range. With this network, the difference between training and validation accuracy decreased to 4-5% as well so increasing the network's size solved the overfitting problem that the smaller lenet network had. With the current design, no dropout is necessary because the data augmentation generator ensures that a portion of the batch is always new data (or at least slightly adjusted). Adding dropout, increases the time to convergence, and with my current network size, actually decreases my accuracy. I assume that with much deeper networks with number of filters in the 100's, the network would overfit (even with augmented data) and at that point dropout would be a necessity.

If a well known architecture was chosen:
* What architecture was chosen?
Lenet was the base but adjustments were made on top of it as stated above.
* Why did you believe it would be relevant to the traffic sign application?
The Lenet was suited for 32x32x3 RGB CIFAR-10 images so the image size and number of channels seemed very similar to the traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final architecture was not exactly Lenet and is explained above.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

After resizing the images, I felt that the predestrian and beware of snow/ice would be harder for the neural network to predict because the black drawings in the middle of the sign are very pixelated and have a shape that does not represent the original insignia. I tried to slowly downsample the images and do gaussian blur on the images but in the end, opencv's resize function performed about the same if not a better job. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14th, 15th and 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice/snow   	| Road narrows on the right  					| 
| No entry     			| No entry 										|
| Pedestrians			| Pedestrians									|
| Road work      		| Road work					 					|
| Speed limit (30km/h)	| Speed limit (30km/h)							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. With only 5 images, and considering the test accuracy of 94.4%, all images should have been identified correctly, but it missed one. To better understand why certain labels performed better than others I looked at the sklearn classification report:

![alt text][image16]

It seems that the "beware of snow/ice" sign has a slightly lower f1-score of 0.71 so it has a higher chance of producing a false positive than "no entry" which has an f1-score of 1.0. "No entry" has a 100% chance of being predicted correctly with its previously mentioned f1-score. What is surprising is that the "pedestrian" sign is predicted correctly as it has a f1-score of 0.28. The "road work" sign is another class that has high f1-score at 0.91 as with the "speed limit (30km/h)" at 0.93. The only anomaly in this case was that a "pedestrian" sign was predicted correctly. It could be that with a small data set, the pedenstrian sign just happened to be predicted correctly, but with a larger test set of more pedestrian signs, the pedestrian signs from the internet would not perform well at all compared to the other sign labels that had higher f1-score.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a beware of ice/snow sign (probability of about 0.66), and the image containa a Road narrows on the right sign. The top five soft max probabilities were

![alt text][image17]

For the second image, the model is very sure that this is a no entry sign (probability of about 0.95), and the image does contain a no entry sign. The top five soft max probabilities were

![alt text][image18]

For the third image, the model is very sure that this is a pedestrian sign (probability of about 0.98), and the image does contain a pedestrian sign. The top five soft max probabilities were

![alt text][image19]

For the fourth image, the model is very sure that this is a road work sign (probability of 0.97), and the image does contain a road work sign. The top five soft max probabilities were

![alt text][image20]

For the fifth image, the model is very sure that this is a speed limit 30km/h (probability of 0.99), and the image does contain a speed limit 30km/h sign. The top five soft max probabilities were

![alt text][image21]

####4. Visualization of network feature maps
The visualization of the neural network's state are show in the ipython notebook starting from cell 18 and on. It seems that the first convolution layer output shows the large features of the image. The overall shape of the sign is included and a good portion of detail on the insignia is shown as well. This makes sense because the small 5x5 filter is running through the entire original image, pixel by pixel. The second convolution layer seems to be focused more on the overall shape of the sign, a triangle. This layer omits the details of the center of the sign and just finds the triangular shape. The last layer seems to show just lines which is not as intuitive but considering that each pixel in the output of this layer is representing a large portion of the original image with just a 0-255 value, this last convolution layer cannot contain much detail about the image. It just provides information about the areas that are usually bright or dark in the original image. This is useful because as seen in a few of the feature maps the lines are diagonals and therefore these layers are learning the orientation of edges in the overall image.