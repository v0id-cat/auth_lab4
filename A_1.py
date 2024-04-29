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




# Setup Training Data
train_dir = './data/TRAIN'
train_labels = []
train_img_names = []
train_img_paths = []
train_gender = []

for subdir, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(subdir, file), 'r') as t:
                content = t.readlines()
                train_gender.append(content[0].rsplit(' ')[1][0])
                img_name = content[2].rsplit(' ')[1][:-4] + '.png'
                train_img_paths.append(os.path.join(subdir, img_name))
                train_img_names.append(img_name)
                train_labels.append((content[1].rsplit(' ')[1][0]))           
train_df = pd.DataFrame()
train_df['IMAGE PATH'] = train_img_paths
train_df['IMAGE NAME'] = train_img_names
train_df['LABEL'] = train_labels
train_df['GENDER'] = train_gender




# Setup Test Data
test_dir = './data/TEST'
test_labels = []
test_img_names = []
test_img_paths = []
test_gender = []

for subdir, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(subdir, file), 'r') as t:
                content = t.readlines()
                test_gender.append(content[0].rsplit(' ')[1][0])
                img_name = content[2].rsplit(' ')[1][:-4] + '.png'
                test_img_paths.append(os.path.join(subdir, img_name))
                test_img_names.append(img_name)
                test_labels.append((content[1].rsplit(' ')[1][0]))           
test_df = pd.DataFrame()
test_df['IMAGE PATH'] = test_img_paths
test_df['IMAGE NAME'] = test_img_names
test_df['LABEL'] = test_labels
test_df['GENDER'] = test_gender



train_df.drop(columns = 'GENDER',inplace=True)
train_classes = list(np.unique(train_labels))
test_classes = list(np.unique(test_labels))
train_map_classes = dict(zip(train_classes, [t for t in range(len(train_classes))]))
test_map_classes = dict(zip(test_classes, [t for t in range(len(test_classes))]))
train_df['MAPPED LABELS'] = [train_map_classes[i] for i in train_df['LABEL']]
test_df['MAPPED LABELS'] = [test_map_classes[i] for i in test_df['LABEL']]
train_df = train_df.sample(frac = 1) #To randomly shuffle the data
test_df = test_df.sample(frac = 1) #To randomly shuffle the data



X_train = train_df['IMAGE PATH']
y_train = train_df['MAPPED LABELS']
X_test = test_df['IMAGE PATH']
y_test = test_df['MAPPED LABELS']

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
        print("asfasfaf")
        continue
# Creating numpy arrays of images
X_train_images = preprocess_input(np.array(X_train_images, dtype=float))
X_test_images = preprocess_input(np.array(X_test_images, dtype=float))

# Convert labels to one-hot format
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=5)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=5)




# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Fine-Tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True


from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

train_generator = train_datagen.flow(X_train_images, y_train_one_hot, batch_size=64)

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
history = model.fit(train_generator, epochs=5, validation_data=(X_test_images, y_test_one_hot), callbacks=[lr_scheduler, early_stopping], verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_images, y_test_one_hot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)







print("end")
