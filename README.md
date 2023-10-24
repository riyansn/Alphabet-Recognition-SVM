Handwritten Alphabet Recognition
This project focuses on recognizing handwritten alphabets using the LinearSVC model from Scikit-learn. The dataset is extracted from a CSV file that contains pixel values of images along with their corresponding labels.

Table of Contents
Define function
Data Preparation
Load Data
Modeling
Flattening The Image
Train Test Splitting
Pipelining
Evaluation
Precision, Recall, F1-score, and Accuracy
Save the Model
Load the Model
Prerequisites
The following libraries and extensions are used:

csv
numpy
os
string
pickle
PIL from Image
sklearn
sklearnex
matplotlib
1. Define function
Three key functions are defined:

csv_to_image: This function loads images from a CSV file and saves them as individual image files in folders categorized by their labels.
load_image: This function loads images from a directory and returns them as a stacked numpy array along with their labels.
csv_to_array: This function loads images from a CSV file and returns them as a stacked numpy array for handwritten recognition.
2. Data Preparation
Images are extracted from the given CSV file and saved as individual image files in their respective folders.

3. Modeling
Flattening The Image
To make the image data suitable for training using LinearSVC, the images are flattened.

Train Test Splitting
The dataset is split into a training set and a testing set with an 80-20 split.

Pipelining
For efficient training and preprocessing, a pipeline is created which first standardizes the data using StandardScaler and then trains it using LinearSVC.

4. Evaluation
The model's predictions on the test data are evaluated using metrics like precision, recall, F1-score, and accuracy.

5. Save the Model
The trained model is saved to the disk using pickle.

6. Load the Model
The saved model can be loaded back for inference or further training.

Note: Make sure to have the required CSV file and folders for image storage in the mentioned paths before running the code.
