from m1 import method1
from m2 import method2
from m3 import method3

import os
import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

def data_setup():
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
	train_df.drop(columns = 'GENDER',inplace=True)
	
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
	test_df.drop(columns = 'GENDER',inplace=True)
		
	train_classes = list(np.unique(train_labels))
	test_classes = list(np.unique(test_labels))
	train_map_classes = dict(zip(train_classes, [t for t in range(len(train_classes))]))
	test_map_classes = dict(zip(test_classes, [t for t in range(len(test_classes))]))
	train_df['MAPPED LABELS'] = [train_map_classes[i] for i in train_df['LABEL']]
	test_df['MAPPED LABELS'] = [test_map_classes[i] for i in test_df['LABEL']]
	train_df = train_df.sample(frac = 1) #To randomly shuffle the data
	test_df = test_df.sample(frac = 1) #To randomly shuffle the data
	
	return train_df, test_df

def hybrid(train, test):
	# for each method, get FRRavg, FRRmin, FRRmax, FARavg. FARmin, FARmax, ERR
	FAR1, FRR1, EER1 = method1(train, test)
	FAR2, FRR2, EER2 = method2(train, test)
	FAR3, FRR3, EER3 = method3(train, test)

	stats = [mean([FRR1, FRR2, FRR3]),
			 min([FRR1, FRR2, FRR3]),
			 max([FRR1, FRR2, FRR3]),
			 mean([FAR1, FAR2, FAR3]),
			 min([FAR1, FAR2, FAR3]),
			 max([FAR1, FAR2, FAR3]),
			 mean([EER1, EER2, EER3])]
			 
	return stats

def main():
	train, test = data_setup()
	stats = hybrid(train, test)
	print (stats)

if __name__ == "__main__":
	main()
