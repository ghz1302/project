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

[image1]: ./examples/visualization.png "Visualization"
[image12]: ./examples/data0.jpg "Data"
[image2]: ./examples/grayscale0.jpg "Grayscaling"
[image3]: ./examples/normal0.jpg "Normalizing"
[image4]: ./NewImages/0.jpg "Traffic Sign 1"
[image5]: ./NewImages/1.jpg "Traffic Sign 2"
[image6]: ./NewImages/2.jpg "Traffic Sign 3"
[image7]: ./NewImages/3.jpg "Traffic Sign 4"
[image8]: ./NewImages/4.jpg "Traffic Sign 5"


---
###Writeup / README

###My [project code](https://github.com/ghz1302/project/blob/master/CarND-Traffic-Sign-Classifier-Project-master/Traffic_Sign_Classifier.ipynb) is here.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I found summary statistics of the traffic signs data set:

| Data set         		|     size	        					|
|:-------------------------:|:--------------:| 
| The size of training set    |      34799      |
| The size of the validation set  |     4410    |
| The size of test set  |    12630    |
| The shape of a traffic sign image  |    (32, 32, 3)    |
| The number of unique classes/labels in the data set    |     43    |

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how preprocessed the image data.

As a first step, I convert the images to grayscale because it makes easier to compute and faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image12]  ![alt text][image2]

As a last step, I normalized the image data because it makes easier for classifier to learn.

Here is an example of normalized image:

![alt text][image3]



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| output = 400        									|
| Fully connected		| output = 120									|
| RELU					|												|
| Dropout					|		probability = 0.75									|
| Fully connected		| output = 10									|
 


####3. Describe how  trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used parameters below

parameter| |
|---|---|
mu | 0
sigma | 0.05
EPOCHS | 20
BATCH_SIZE | 512
learning rate | 0.001

and results were

set | |
|---|---|
training set accuracy  | 0.984
validation set accuracy  | 0.884
test set accuracy | 0.874

It didn't reach the goal ' Validation accuracy 0.93', so I tried different number of parameters. 
I used lower learning rate and more epochs because it could make loss lower.
And smaller batches sizes could have better accuracy, so I used smaller batch sizes.

parameter| |
|---|---|
mu | 0
sigma | 0.078
BATCH_SIZE | 192
learning rate | 0.0009

At EPOCH 42, I got results :

set | |
|---|---|
training set accuracy  | 1.000
validation set accuracy  | 0.932
test set accuracy | 0.914


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)   									| 
| Road work     			| Road work 										|
| Priority road					| Priority road											|
| Turn left ahead	      		| Turn left ahead					 				|
| Speed limit (60km/h)			| Speed limit (60km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The top five soft max probabilities of using parameters about validation set accuracy 0.884
For the first image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (70km/h)   									| 
| .001     				| Speed limit (30km/h) 										|
| .0003					| Priority road											|
| .0003	      			| Speed limit (120km/h)					 				|
| .00005				    | Speed limit (50km/h)      							|


For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Road work   									| 
| .000004     				| Wild animals crossing 										|
| .000002					| Beware of ice/snow											|
| .000002	      			| Slippery road					 				|
| .0000001				    | Road narrows on the right     							|

For the third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road   									| 
| .001     				| No entry 										|
| .0004					| Speed limit (30km/h)										|
| .0002	      			| Speed limit (50km/h)			 				|
| .0001				    | Stop      							|

For the fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Turn left ahead   									| 
| .000007     				| Keep right										|
| .00000001				| Speed limit (60km/h)										|
| .0000000001	      			| No vehicles			 				|
| .000000000005				    | Go straight or right   							|

For the fifth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (60km/h)  									| 
| .0004     				| Speed limit (50km/h)									|
| .00006					| Speed limit (80km/h)									|
| .000005	      			| Ahead only 			 				|
| .0000008				    | Speed limit (30km/h)     							|

After, I tried parameters about validation set accuracy 0.932.
I got almost 1.00 probability about all of 5 traffic signs.
