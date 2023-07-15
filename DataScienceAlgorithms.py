import numpy as np
import pandas as pd
from collections import Counter

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
        :return: Tuple of numpy vectors
        """
        X = self.df.values[:, :4].astype(float)  # all rows (:), columns 0 -> 3 (:4)
        y = self.df.values[:, 4]
        mask = np.array([True] * 120 + [False] * 30)

        return self.raw_data

# Optimized implementation of 'KNearestNeighbors' algorithm using numpy
class KNearestNeighbors():

    def __init__(self, k, distance_metric="euclidean"):
        """ Custom class to implement the k-NearestNeighbors classification algorithm through the Numpy package
        :param k:
        :param distance_metric:
        """
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Prepare the broadcast
        self.X_train_reshaped = np.expand_dims(self.X_train, 1) # Data prepared for the "eucldean distance" distance implementation
        self.X_train_norm = ((self.X_train ** 2).sum(axis=1) ** .5).reshape(-1, 1)

    def euclidean(self, X_test):
        X_diff = self.X_train_reshaped - X_test
        dist_matrix = ((X_diff ** 2).sum(axis=2)) ** .5
        return dist_matrix

    def majority_voting(self, votes):
        """ Naive implementation to assign a label
        :param votes: Set of labels (string object in this case)
        :return: More common object
        """
        count = Counter(votes)
        return count.most_common(1)[0][0]

    def predict(self, X_test):
        """ Use the train set to label the test data points

        :param X_test: Data points to label
        :return: Predicted labels
        """
        if self.distance_metric == "euclidean":
            dist_matrix = self.euclidean(X_test)
        # For every data point, select the smallest data point in the original train_set (considering the 'idx')
        knn = dist_matrix.argsort(axis=0)[:self.k, :].T # Transpose the matrix for the final labeling
        y_pred = np.array([self.majority_voting(self.y_train[knn][i]) for i in range(len(self.y_train[knn]))])
        return y_pred



# Implement the custom class
def main():
    df = DatasetWrapper("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    df.preliminary_exploration()
    knn_model = KNearestNeighbors(3, "euclidean")
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

if __name__ == '__main__':
    main()