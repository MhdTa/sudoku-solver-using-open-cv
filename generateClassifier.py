# Import the modules
import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import numpy as np
from collections import Counter
import os
import cv2

# Load the dataset
print('loading..')
dataset = datasets.fetch_openml('mnist_784')
joblib.dump(dataset, "dataset.pkl", compress=3)

dataset = joblib.load("dataset.pkl")

print('done')
# print(dataset)

# Extract the features and labels
# features = np.array(dataset.data, 'int16')
# labels = np.array(dataset.target, 'int')
# print(len(features))

# Extract the hog features
# print('Extracting..')
# list_hog_fd = []
# for feature in features:
#     fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),)
#     list_hog_fd.append(fd)
# hog_features = np.array(list_hog_fd, 'float64')
# #
# joblib.dump(hog_features, "hog_features.pkl", compress=3)

# hog_features = joblib.load("hog_features.pkl")

# print("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
# clf = SVR()

winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
# compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (8, 8)
padding = (8, 8)
locations = ((10, 20),)

path = 'C:\\Users\Omar\Desktop\sampels'
list_hog_fd = []
files = os.listdir(path)
trains = []
labels = []
# hog = cv2.init_hog_descripter()

# hog = cv2.HOGDescriptor()
# im = cv2.imread(sample)


for file in files:
    image = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
    # image = np.array(image,'int16' )
    # fd = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),)
    # list_hog_fd.append(fd)
    # hog_features = np.array(list_hog_fd, 'float64')
    # hist = cv2.hog_compute(hog, image)
    # print(image.shape)
    hist = hog.compute(image, winStride, padding, locations)
    print(hist.shape)
    hist = np.reshape(hist, (-1))
    # print(hist.shape)
    trains.append(hist)
    labels.append((int(file.split("-")[0])))
# print(hist.shape)
trains = np.matrix(trains, dtype=np.float32)
print(trains.shape)
labels = np.array(labels)
# labels.resize((labels.shape[0], 1))
print(labels)

clf = LinearSVC()

# Perform the training
print('training..')
clf.fit(trains, labels)

# Save the classifier
joblib.dump(clf, "svc1.pkl", compress=3)
