import struct
import numpy as np
from collections import Counter, defaultdict
import itertools
from math import sqrt
import os
import copy

TRAIN_FILE = "train-images.idx3-ubyte"
TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"
DATA_DIR = "./data/"


class Datapreprocess(object):
    def __init__(self, data_dir):
        self.root_dir = data_dir

    def parse_labels(self, data):
        """Parse labels from the binary file. Reference: stack overflow."""

        magic, n = struct.unpack_from('>2i', data.read(8))
        assert magic == 2049, "Wrong magic number: %d" % magic

        # Next, let's extract the labels.
        labels = struct.unpack_from('>%dB' % n, data.read())
        return labels

    def parse_images(self, data):
        """Parse images from the binary file.
        Reference: stack overflow."""

        # Parse metadata.

        magic, n, rows, cols = struct.unpack_from('>4i', data.read(16))
        assert magic == 2051, "Wrong magic number: %d" % magic

        # Get all the pixel intensity values.
        num_pixels = n * rows * cols
        pixels = struct.unpack_from('>%dB' % num_pixels, data.read())

        # Convert this data to a NumPy array for ease of use.
        pixels = np.asarray(pixels, dtype=np.float)

        # Reshape into actual images instead of a 1-D array of pixels.
        images = pixels.reshape((n, cols, rows))
        return images

    def get_mnist_data(self):
        parse_train_images = self.parse_images(open(os.path.join(self.root_dir, TRAIN_FILE), "rb"))
        parse_train_labels = self.parse_labels(open(os.path.join(self.root_dir, TRAIN_LABEL_FILE), "rb"))
        parse_test_images = self.parse_images(open(os.path.join(self.root_dir, TEST_FILE), "rb"))
        parse_test_labels = self.parse_labels(open(os.path.join(self.root_dir, TEST_LABEL_FILE), "rb"))
        return parse_train_images, parse_train_labels, parse_test_images, parse_test_labels


def get_cropped_img(m):
    n = np.pad(m, 1, mode='constant')
    r, c = len(m), len(m[0])
    for x in range(3):
        for y in range(3):
            yield n[x:x + r, y:y + c]


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


class KNN:
    def __init__(self, train_data, k):
        self.train = train_data
        self.k = k

    def predict(self, test):
        """Return the k-nearest neighbor prediction for test image."""
        distances = [(distance(x[0], test), index) for index, x in enumerate(self.train)]

        sorted_dist = sorted(distances)

        labels = []
        for nbr in range(self.k):
            index = sorted_dist[nbr][1]
            labels.append(self.train[index][1])

        prediction = Counter(labels).most_common(1)[0][0]

        return prediction

    def slide_predict(self, test):
        distances = []
        for index, x in enumerate(self.train):
            slide_dis = [distance(y, test) for y in get_cropped_img(x[0])]
            distances.append((min(slide_dis), index))
        sorted_dist = sorted(distances)
        labels = []
        for nbr in range(self.k):
            index = sorted_dist[nbr][1]
            labels.append(self.train[index][1])
        prediction = Counter(labels).most_common(1)[0][0]
        return prediction


def print_table(table):
    """
    This method is just to format the edit distance and backtrace tables.
    :param table: edit distance or back trace table
    :param source: source sequence
    :param target: target sequence
    :return: formatted tables
    """
    matrix = copy.deepcopy(table)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    second_row = [' ', 'act|pred'] + labels

    for ind, row in enumerate(matrix):
        row.insert(0, ind)
        row.insert(0, ' ')
    matrix.insert(0, second_row)

    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(rows):
        for col in range(cols):
            label = str(matrix[row][col])
            element = label
            print('{0}'.format(element).center(7), end='  ')
        print()


print("Creating data object to load MNIST data files...")
data_object = Datapreprocess(DATA_DIR)
print("Extracting data..")
train_images, train_labels, test_images, test_labels = data_object.get_mnist_data()

print("***************Data Dimensions****************")
print("train_images shape:", train_images.shape)
print("train_labels shape:", len(train_labels))
print("test_images shape:", test_images.shape)
print("test_labels shape:", len(test_labels))

# Create a zip of train images and labels
dataset = []
for i in range(len(train_images)):
    dataset.append((train_images[i, :, :], train_labels[i]))

# Shuffle the data and divide it into 10 folds
print("shuffling the data and dividing it into ten folds of equal size..")
shuffle_data = dataset[:]
np.random.shuffle(shuffle_data)
folds = []
for i in range(0, len(shuffle_data), 6000):
    folds.append(shuffle_data[i:i + 6000])

# 10 fold cross validation
print("Calculating optimal k value, Running KNN for 10 folding cross validation on train_data..")
k_scores = []
fold_scores_per_k = defaultdict(list)
for k in range(1, 11):
    fold_scores = []
    for i in range(len(folds)):
        train_data = folds[:]
        del train_data[i]

        train_data = list(itertools.chain.from_iterable(train_data))
        test_data = folds[i]

        print("Running for k value:", k, "fold number: ", i, "for train data size:", len(train_data),
              "validation data size:", len(test_data))
        model = KNN(train_data, k)
        predictions = [model.predict(test_data[i][0])
                       for i in range(len(test_data))]
        actuals = [label[1] for label in test_data]
        total_correct = np.sum(np.asarray(predictions) == np.asarray(actuals))
        total_preds = len(predictions)
        fold_scores.append(total_correct / total_preds)
    fold_scores_per_k[k].extend(fold_scores)
    k_scores.append(sum(fold_scores) / len(folds))
    print("fold scores for k=",k,":", fold_scores_per_k[k])
    print("k scores",k_scores)
    print("accuracy for k value ", k, " is ", k_scores[-1])

optimal_k = k_scores.index(max(k_scores)) + 1

print("optimal k value:", optimal_k)

# Run knn classifier
print("Running KNN on train_data and test_data of MNIST..")
train_model = KNN(dataset, optimal_k)
actual_labels = np.asarray(test_labels)
test_predictions = [train_model.predict(test_images[i, :, :]) for i in range(len(test_images))]
pred_labels = np.asarray(test_predictions)

# find error and accuracy of knn classifier
print("calculating accuracy and error of knn..")
total_correct_preds = np.sum(np.asarray(test_predictions) == np.asarray(actual_labels))
test_accuracy = total_correct_preds / len(test_predictions)
total_incorrect_preds = np.sum(np.asarray(test_predictions) != np.asarray(actual_labels))
test_error = total_incorrect_preds / len(test_predictions)

print("test_accuracy:", test_accuracy*100)
print("test_error:", test_error*100)

# construct and print confusion matrix
print("print confusion matrix")
confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
for actual, predicted in zip(actual_labels, pred_labels):
    confusion_matrix[actual][predicted] += 1
print_table(confusion_matrix)

# Confidence interval for knn classifier
print("confidence interval with z val 1.96..")
conf_val = 1.96 * sqrt((test_accuracy * (1 - test_accuracy)) / len(test_labels))
conf_interval = (test_accuracy - conf_val, test_accuracy + conf_val)
print("confidence interval:", conf_interval)

"""
Sliding window knn classifier
"""
print("Running sliding window knn classifier..")
sliding_train_model = KNN(dataset, optimal_k)
test_slide_predictions = [sliding_train_model.slide_predict(test_images[i, :, :]) for i in range(len(test_images))]

"""
calculate error and accuracy of knn classifier
"""
print("calculating accuracy and error of sliding window knn classifier..")
total_slide_crct_preds = np.sum(np.asarray(test_slide_predictions) == np.asarray(actual_labels))
test_accuracy_slide = total_slide_crct_preds / len(test_slide_predictions)
total_incorrect_preds_slide = np.sum(np.asarray(test_slide_predictions) != np.asarray(actual_labels))
test_error_slide = total_incorrect_preds_slide / len(test_slide_predictions)

print("test_slide_accuracy:", test_accuracy_slide*100)
print("test_slide_error:", test_error_slide*100)

"""
Confusion matrix for sliding window knn
"""
print("printing confusion matrix for sliding window..")
pred_labels_slide = np.asarray(test_slide_predictions)
confusion_matrix_slide = [[0 for _ in range(10)] for _ in range(10)]
for actual, predicted in zip(actual_labels, pred_labels_slide):
    confusion_matrix_slide[actual][predicted] += 1

print_table(confusion_matrix_slide)

"""
Confidence interval for sliding window knn classifier
"""
print("confidence interval fr sliding window with z value 1.96..")
conf_val_slide = 1.96 * sqrt((test_accuracy_slide * (1 - test_accuracy_slide)) / len(actual_labels))
conf_interval_slide = (test_accuracy_slide - conf_val_slide, test_accuracy_slide + conf_val_slide)
print("confidence interval sliding window:", conf_interval_slide)
