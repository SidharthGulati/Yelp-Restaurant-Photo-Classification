# Yelp-Restaurant-Photo-Classification
Source Code for Yelp Restaurant Photo Classification using CNN (transfer learning)
Methodology :

a) The bottleneck values (CNN codes) are generated by randomly sampling batches of 300 images. Along with this, one hot encoded vector for the image labels is also generated and stored with the same batch size. These bottleneck values and one-hot encoded vectors are stored as .npy files. This process is repeated for all the images in the dataset with a sample size of 300. Finally we get 782 .npy files for CNN codes and 782 .npy files for the one-hot encoded labels. The bottleneck values are stored in bottleneck folder and img labels are stored in img_label in the cwd.

b) These 782 .npy files are then are then concatenated (both bottleneck values and image labels ) and are fed to the Neural Network as input to the Main.py. In Main.py, training of the last classification layer is done and then it generate the predicted labels of the test set for each of the nine labels and find the cross validation accuracy, training accuarcy and cross-entropy loss values.

c) The predicted labels and ground truth labels are stored separately and are used to obtain the F1-score,precision , recall values and other graphs.

To run the project, follow the below steps:

Step 1) Execute Generate_Bottleneck.py.

Step 2) Then, execute Main.py.