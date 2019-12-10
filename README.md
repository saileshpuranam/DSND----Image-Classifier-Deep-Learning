
# Image Classifier Project - Deep Learning with PyTorch

### Project Motivation and Objective

In this project, we'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. We will train this classifier, then export it for use in our application. We'll be using this dataset of 102 flower categories.

I used torchvision to load the data. The dataset is split into three parts, training, validation, and testing. For the training, applied transformations such as random scaling, cropping, and flipping based on specification given by Udacity. This will help the network generalize leading to better performance. Also need to load in a mapping from category label to category name in json format. I then created classification after training and testing the model. After training and testing the model, we have used this to classify and predict images along with associated probabilities.

I achieved an accuracy of 89% on training dataset and 91% with the test dataset. The resultant model also rightly predicted the image when we applied this model to calsssify images

The project is broken down into below steps:

1. Load and transform the image dataset.
2. Train the image classifier model based on the training dataset.
3. Based on the trained classifier model, predict the flower names and probabilities based on image content.
4. Create train and predict modules so that any image dataset can be predicted using these modules


### Datasets Used

In this project, I used a dataset of 102 flower categories, images and names

### Python Packages used

This process requires Python 3 and above for efficient execution

This project is executed in a Jupyter Notebook with the help of following packages - 
Torch, PIL, Matplotlib.pyplot, Numpy, Scikit-learn, Seaborn, Torchvision, Json and time. 

### Jupyter Notebook

Jupyter Notebook used for this analysis is uploaded in the repository titled - **Image Classifier Project.ipynb**

All the steps incolved in loading, transforming and training the datasets & Predicting using Pytorch models are clearly explained in the notebook with comments.

### Command Line Application

#### Train the dataset using train.py module

All the functions and codes required for training the dataset are available in train.py file. Below are the steps to run it on Command line application. Train modules generates a checkpoint with model features and parameters which would be used for prediction

1. **Set up the file directory path -** cd /home/workspace/ImageClassifier
2. **Call train.py module and pass the arguments required for the module to run -** python train.py --save_dir “./home/workspace/ImageClassifier” --learning_rate 0.001 --hidden_units 4096 --epochs 5 –gpu 


#### Use the trained checkpoint to predict and classify images

All the functions and codes required for prediction are available in predict.py file. Below are the steps to run it on Command line application

1. **Set up the file directory path -** cd /home/workspace/ImageClassifier
2. **Call train.py module and pass the arguments required for the module to run -** python predict.py --image "./flowers/test/10/image_07117.jpg" --checkpoint my_checkpoint.pth --top_k 5  --category_names cat_to_name.json --gpu

### Github Repo Contents

1. Image Classifier model.ipynb 
2. Category to names json file
3. train.py and predict.py modules used on command line
4. Project README file

### Acknowledgements and Licensing

Datatsets for this project are exctracted through the Udacity course 
Feel free use the code and other info in this repo for all your references whenever required.
