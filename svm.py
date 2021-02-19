import glob

import cv2
import joblib
import numpy as np
from skimage.feature import hog as hog_image
from sklearn.svm import SVC

hog = cv2.HOGDescriptor()
labels = []
test_data = []
test_labels = []

def generate_hog_vectors(path, label, purpos):
    data = []
    for file_name in glob.iglob(path, recursive=True):
        img = hog_image(file_name)[0]
        if purpos == 'train':
            data.append(img)
            labels.append(label)
        #  print(len(labels))
        else:
            # print(len(test_data))
            # print(len(test_labels))
            data.append(img)
            test_labels.append(label)
    return data


def generate_classifier(hog_vectors, labels):
    labels = np.array(labels)
    classifier = SVC(C=10000, kernel="linear", gamma=0.000001)
    classifier.fit(hog_vectors, labels)
    return classifier


def structur_list(input_list):
    return np.stack(input_list)


def hog_image(path):
    image = cv2.imread(path, 0)
    dim = 128
    img = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_AREA)
    img = hog.compute(img)
    img = np.squeeze(img)
    print(type(img))
    return [img]


check = False

joblib_file = "pedestrian_detection_model.pkl"

'''def creat_test_data():

    test_data=[]
    test_data = generate_hog_vectors('/home/kader93t/Desktop/data/validation/no pedestrian/*.jpg', 0, 'test')
    test_data += generate_hog_vectors('/home/kader93t/Desktop/data/validation/pedestrian/*.jpg', 1, 'test')
    test_data = structur_list(test_data)
'''


def load_data(path1, path2, purpos):
    if purpos == 'test':
        test_data.clear()
        test_labels.clear()
    data = []
    data.clear()
    data = generate_hog_vectors(path2, 0, purpos)
    data += generate_hog_vectors(path1, 1, purpos)
    return data


# data = load_data('/home/kader93t/Desktop/data/train/pedestrian/*.*','/home/kader93t/Desktop/data/train/no pedestrian/*.jpg')

def get_model(check, file, data):
    # creat_test_data()
    joblib_file = file
    test_data = test_labels = []
    if not check:
        print(type(data))
        data = structur_list(data)
        classifier = generate_classifier(data, labels)
        # joblib.dump(classifier, joblib_file)
    else:
        classifier = joblib.load(joblib_file)
    return classifier
# print(get_model(False,'',data))
# print(get_model(True,joblib_file).predict(list_to_matrix( hog_image('/home/kader93t/Desktop/data/test/images.jpeg'))))
