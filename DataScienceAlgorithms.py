import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from math import floor
from sklearn.datasets import make_blobs

# For the reproducibilty of the experiments
np.random.seed(10)

# Utils function
def check_variable_value(variable, value_list):
    """
    :param variable: The variable to check
    :param value_list: List of values allowed
    :return: True if the variable is correctly set
    """
    if not isinstance(variable, str):
        raise TypeError("This parameter should be a string.")

    if variable not in value_list:
        raise ValueError("Variable value not found in the list.")

    return True

def accuracy_score(y_true, y_pred):
    return (y_true==y_pred).sum()/len(y_true)

class DatasetWrapper():

    def __init__(self, link):
        """ Classes of utilities for the dataset processing (implementing Pandas and Numpy packages)

        :param link: Link to use for downloading the dataset
        """
        self.link_dataset = link
        self.df = pd.read_csv(link, header=None)
        self.raw_data = self.df.values

    def preliminary_exploration(self):
        """ Call Pandas functions to implement the general data exploration phase

        :return: None
        """
        print("***   Data exploration   ***\n\nFirst few rows:")
        print(self.df.head())

        # Check the dimensions of the DataFrame
        print("\nShape of the DataFrame:")
        print(f"{self.df.shape}")

        # Get the column names
        print("\nColumn names:")
        print(f"{self.df.columns}")

        # Summary statistics of numerical columns
        print("\nSummary statistics:")
        print(f"{self.df.describe()}")

        # Count missing values in each column
        print(f"\nMissing values:")
        print(f"{self.df.isnull().sum()}")

        return None

    def get_split_data(self, split_percentage=0.8):
        """ Function to retrieve the Train, Test split using the 'mask'
        :param split_percentage: Split of dataset to be used as training set.
        :return: Numpy vectors (X/y split) returned as tuple
        """
        X = self.df.values[:, :4].astype(float)
        y = self.df.values[:, 4]
        train_num = floor(X.shape[0] * 0.8) # Apply the percentage split
        mask = np.array([True] * train_num + [False] * (X.shape[0] - train_num))
        np.random.shuffle(mask)

        # Retrieve the splits
        X_train, X_test = X[mask], X[~mask]
        y_train, y_test = y[mask], y[~mask]

        return (X_train, X_test, y_train, y_test)

# Implementation of 'KNearestNeighbors' algorithm using numpy
class KNearestNeighbors():

    def __init__(self, k, distance_metric="euclidean", weights="uniform"):
        """ Custom class to implement the k-NearestNeighbors classification algorithm through the Numpy package
        :param k:
        :param distance_metric:
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weight_mode = weights
        self.allowed_weight_modes = ["uniform", "weighted"]
        self.allowed_distance_metrics = ["euclidean", "cosine"]
        # Checking integrity of the paramters
        check_variable_value(self.distance_metric, self.allowed_distance_metrics)
        check_variable_value(self.weight_mode, self.allowed_weight_modes)

    def fit(self, X_train, y_train):
        """ Store and prepare the dataset to predict unlabeled data points
        :param X_train: Training features of the dataset
        :param y_train: Set of labels for each data point
        :return: None
        """
        self.X_train = X_train
        self.y_train = y_train
        # Prepare the broadcast for differents similarities fuction
        self.X_train_reshaped = np.expand_dims(self.X_train, 1) # Data prepared for the "eucldean distance" distance implementation
        self.X_train_norm = ((self.X_train ** 2).sum(axis=1) ** .5).reshape(-1, 1)

        return None

    def euclidean(self, X_test):
        """ Compute the euclidean distance
        :param X_test: Features from the unlabeled data points
        :return: Matrix of the distance along the all dimension for each pair of train and test data point
        """
        X_diff = self.X_train_reshaped - X_test
        dist_matrix = ((X_diff ** 2).sum(axis=2)) ** .5
        return dist_matrix

    def cosine(self, X_test):
        """ Compute the cosine distance
        :param X_test: Features from the unlabeled data points
        :return: Matrix of the distance along the all dimension for each pair of train and test data point
        """
        X_test_norm = ((X_test ** 2).sum(axis=1) ** .5).T
        dot_prods = self.X_train @ X_test.T # Matrix multiplication
        dist_matrix = 1 - abs(dot_prods / self.X_train_norm.reshape(-1, 1) / X_test_norm)
        return dist_matrix

    def majority_voting(self, votes):
        """ Naive implementation to assign a label
        :param votes: Set of labels
        :return: More common object
        """
        count = Counter(votes)
        return count.most_common(1)[0][0]

    def weighted_majority_voting(self, votes, weights):
        """ Weighted voting system based on the distance
        :param votes: Set of labels
        :param weights: Weights of the labels to consider
        :return: The label based on distances from the unlabeled point
        """
        count = defaultdict(lambda: 0)
        for vote, weight in zip(votes, weights):
            count[vote] += weight
        return max(count.items(), key=lambda x: x[1])[0]  # return the max value (use a custom key extractor)

    def predict(self, X_test):
        """ Use the train set to label the test data points

        :param X_test: Data points to label
        :return: Predicted labels as an array of values
        """
        if self.distance_metric == "euclidean":
            dist_matrix = self.euclidean(X_test)
        elif self.distance_metric == "cosine":
            dist_matrix = self.cosine(X_test)

        # For every data point, select the smallest data point in the original train_set (considering the 'idx')
        knn = dist_matrix.argsort(axis=0)[:self.k, :].T # Transpose the matrix for the final labeling (final shape of the idx is Nxk)

        if self.weight_mode == "uniform":
            y_pred = np.array([self.majority_voting(self.y_train[knn][i]) for i in range(len(self.y_train[knn]))])
        elif self.weight_mode == "weighted":
            weights = 1 / (np.take_along_axis(dist_matrix, knn.T, 0) + 1e-5) # Add a small value to counter the "0" denominator
            y_pred = np.array([self.weighted_majority_voting(self.y_train[knn][i], weights[:, i]) for i in range(len(self.y_train[knn]))])
        return y_pred

class KMeans():

    def __init__(self, n_clusters=3, max_iter = 100):
        """ Custom class to implement the kMeans clustering technique through the Numpy package
        
        :param n_clusters: Number of clusters
        :param max_iter: Maximum number of iteration allowed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

# Implement the custom class of the classificator with a toy dataset
def classification_algorithm():
    df = DatasetWrapper("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    df.preliminary_exploration()
    X_train, X_test, y_train, y_test = df.get_split_data()

    # List of parameter K
    k_values = list(range(2, 15))
    results = [] # List to append the tuple in form of (k, accuracy value)
    # Optimization loop
    for k in k_values:
      knn_model = KNearestNeighbors(k, "euclidean")
      knn_model.fit(X_train, y_train)
      y_pred = knn_model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      results.append((k, accuracy))

    #print final results for the train/test splits
    print(f"\nThe classification accuracy for K values : {k_values}\n{results}")

    best_k, accuracy = sorted(results, key=lambda x: x[1], reverse=True) [0]
    print(f"\nThe best K configuration is K: {best_k}   Accuracy: {round(accuracy,3)}")

def clustering_algorithm():
  X = make_blobs(n_samples=1000, n_features=2)


if __name__ == '__main__':
    classification_algorithm()


























