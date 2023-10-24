Handwritten Recognition with Support Vector Machine (SVM)
This project is focused on building a handwriting recognition model using SVM. We will be using the A_Z dataset for training and testing our model.

Table of Contents
Define function
Data Preparation
Load Data
Modeling
Flattening The Image
Train Test Splitting
Pipelining
Evaluation
Precision, Recall, F1-score and Accuracy
Save the model
Load the model
Define function <a name="define-function"></a>
We've defined three main functions:

csv_to_image: This function converts a CSV containing images to an actual image saved in a folder.
load_image: Loads images from a directory and stacks them into a numpy array.
csv_to_array: Instead of saving images, this directly loads images from a CSV and stacks them into a numpy array.
Data Preparation <a name="data-preparation"></a>
Load Data <a name="load-data"></a>
We've taken the dataset from a CSV format and converted it into actual images. Additionally, the dataset can also be loaded directly into a numpy array.

Modeling <a name="modeling"></a>
The images were first flattened to fit the SVM model. The dataset was then divided into training and testing sets. We utilized the Intel Extension for scikit-learn for faster execution.

Flattening The Image <a name="flattening-the-image"></a>
The images needed to be flattened (from 2D to 1D) before feeding them into the SVM model.

Train Test Splitting <a name="train-test-splitting"></a>
The dataset was divided into an 80-20 split for training and testing respectively.

Pipelining <a name="pipelining"></a>
A pipeline was created, comprising of scaling and the SVM classifier.

Evaluation <a name="evaluation"></a>
The model was evaluated using various metrics such as precision, recall, f1-score, and accuracy.

Precision, Recall, F1-score and Accuracy <a name="precision-recall-f1-score-and-accuracy"></a>
The SVM model provided an accuracy of approximately 85%.

Save the model <a name="save-the-model"></a>
Post training, the model was saved onto the disk using the pickle module.

Load the model <a name="load-the-model"></a>
The saved model can be loaded from the disk for future predictions.

