import cv2
import os
import numpy as np


def init_hog_descripter():
    winSize = (16, 16)
    blockSize = (4, 8)
    blockStride = (1, 2)
    cellSize = (4, 8)
    nbins = 9

    derivAperture = 20
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    return hog


def hog_compute(hog, image):
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (1, 1)
    padding = (3, 3)
    locations = ((8, 8),)
    return hog.compute(image, winStride, padding, locations)




def get_train_data(path):
    files = os.listdir(path)
    trains = []
    labels = []
    for file in files:
        image = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        hog = init_hog_descripter()
        hist = hog_compute(hog, image)

        hist = np.reshape(hist, (-1))
        print(hist.shape)
        trains.append(hist)
        labels.append((int(file.split("-")[0])))

    # print(hist.shape)
    trains = np.matrix(trains, dtype=np.float32)
    # print(trains.shape)
    labels = np.array(labels)
    return trains, labels


def train_svm_model(data_path):
    trains, labels = get_train_data(data_path)
    saved = False
    if not saved:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        # svm.setC(2.67)
        # svm.setGamma(5.383)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        # tr = np.matrix([t for t in trains],dtype=np.float32)
        svm.train(trains, cv2.ml.ROW_SAMPLE, labels)
        svm.save('svm_train-test.xml')
    else:
        svm = cv2.ml.SVM_load('svm_train-test.xml')
    print("Train finished")
    return svm

def predict_digit(svm_model,hog, image):
    hist = hog_compute(hog, image)
    hist = np.reshape(hist, (-1))
    hist = np.matrix([hist]).astype(np.float32)
    retval, results = svm_model.predict(hist)
    # results = svm_model.predict(hist)
    return results[0][0]

path = 'C:\\Users\Omar\Desktop\sampels'
# train_svm_model(path)
