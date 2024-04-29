import os
import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def cleanup_img(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    equ = cv2.equalizeHist(img)
    return(cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11))
    
def make_arrays(paths, labels):
	X = []
	Y = []
	Z = []
	
	# Initialize SIFT detector
	sift = cv2.SIFT_create()

	# Loop through each file path and extract features
	for file_path, label in zip(paths, labels):
		# Load the image
		image = cleanup_img(file_path)
		
		# Detect keypoints and compute descriptors
		keypoints, descriptors = sift.detectAndCompute(image, None)  

		# Select 20 descriptors 
		features = descriptors.flatten()[np.random.randint(descriptors.shape[0], size=20)] 

		# Append the features to the feature vectors list
		X.append(features)

		# Append the label to the labels list
		Y.append(label)
		Z.append(image)
	
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	
	return X, Y, Z

def method1(df_train, df_test):
	print("Method 1...")
	# Extract file paths and corresponding labels
	train_paths = df_train['IMAGE PATH']
	test_paths = df_test['IMAGE PATH']
	train_labels = df_train['MAPPED LABELS']
	test_labels = df_test['MAPPED LABELS']

	X_train, Y_train, Z_train = make_arrays(train_paths, train_labels)
	X_test, Y_test, Z_test = make_arrays(test_paths, test_labels)

	# Initialize SVM classifier
	svm_classifier = SVC(kernel='linear', C=1.0)

	# Train the SVM classifier
	svm_classifier.fit(X_train, Y_train)

	# Predict the labels for test data
	Y_pred = svm_classifier.predict(X_test)

	# Calculate accuracy
	accuracy = accuracy_score(Y_test, Y_pred)
	confusion = confusion_matrix(Y_test, Y_pred)

	disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=svm_classifier.classes_)
	disp.plot()
	#plt.show()

	FPm = confusion.sum(axis=0) - np.diag(confusion)
	FNm = confusion.sum(axis=1) - np.diag(confusion)
	TPm = np.diag(confusion)
	TNm = confusion.sum() - (FPm + FNm + TPm)

	FP = FN = TP = TN = 0
	for i in FPm:
		FP += i
	for i in FNm:
		FN += i
	for i in TPm:
		TP += i
	for i in TNm:
		TN += i
	FAR = FP/(TN + FP)
	FRR = FN/(TP + FN)
	ERR = (FAR+FRR)/2
	print("FAR:", FAR)
	print("FRR:", FRR)
	print("EER:", ERR)
	print("Test accuracy:", accuracy)
	
	return FAR, FRR, ERR

