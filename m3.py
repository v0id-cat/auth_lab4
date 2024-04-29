import os
import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def method3(df_train, df_test):
	X_train = df_train['IMAGE PATH']
	Y_train = df_train['MAPPED LABELS']
	X_test = df_test['IMAGE PATH']
	Y_test = df_test['MAPPED LABELS']

	# Define image size
	img_size = (224, 224)  # Resize all images to (224, 224)

	# Creating numpy arrays of images
	X_train_images = []
	X_test_images = []
	for filename in X_train:
		img=cv2.imread(filename)
		try:
		    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
		    height, width , layers = img.shape
		    X_train_images.append(img)
		except:
		    continue

	for filename in X_test:
		img=cv2.imread(filename)
		try:
		    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
		    height, width , layers = img.shape
		    size=(width,height)
		    X_test_images.append(img)
		except:
		    print("error")
		    continue

	# Creating numpy arrays of images
	X_train_images = preprocess_input(np.array(X_train_images, dtype=float))
	X_test_images = preprocess_input(np.array(X_test_images, dtype=float))

	# Convert labels to one-hot format
	Y_train_one_hot = keras.utils.to_categorical(Y_train, num_classes=5)
	Y_test_one_hot = keras.utils.to_categorical(Y_test, num_classes=5)

	# Load the pre-trained ResNet50 model without the top classification layer
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Freeze the weights of the pre-trained layers so they are not updated during training
	for layer in base_model.layers:
		layer.trainable = False

	# Fine-Tuning
	for layer in base_model.layers[-10:]:
		layer.trainable = True

	# Data Augmentation
	train_datagen = ImageDataGenerator(
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		preprocessing_function=preprocess_input
	)

	train_generator = train_datagen.flow(X_train_images, Y_train_one_hot, batch_size=64)

	# Add Dropout to the classification head
	x = base_model.output
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Dense(1024, activation='relu')(x)
	x = keras.layers.Dropout(0.5)(x)  # Add dropout
	predictions = keras.layers.Dense(5, activation='softmax')(x)

	# Create the model
	model = Model(inputs=base_model.input, outputs=predictions)

	# Learning Rate Scheduler
	lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001)

	# Compile the model with regularization
	model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

	# Add early stopping
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

	# Train the model with data augmentation and regularization
	history = model.fit(train_generator, epochs=5, validation_data=(X_test_images, Y_test_one_hot), callbacks=[lr_scheduler, early_stopping], verbose=1)

	# Evaluate the model
	loss, accuracy = model.evaluate(X_test_images, Y_test_one_hot)
	print("Test Loss:", loss)
	print("Test Accuracy:", accuracy)

	# Calculate accuracy
	Y_pred = model.predict(X_test_images)	
	Y_pred = np.argmax(Y_pred, axis=1)
	confusion = confusion_matrix(Y_test, Y_pred)

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
	
