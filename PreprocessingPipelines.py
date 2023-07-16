import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os 

# Utils functions
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

# Get the train/val loader fromt the custom class
def get_loader():
  pass

class ImagesDataset(Dataset):
  """ Class to work with CIFAR-10 (train split)
  """

  def __init__(self, name='CIFAR-10', root_dir='Datasets'):
    """ Custom class for dataset loading (download from Torchvision)

    :param name: Name (str) of the dataset to donwload
    :param root_dir: Path (str) to use when saving the dataset
    """

    if os.path.exists(root_dir):
      print("Downloading the dataset in the folder")
    else:
      print(f"Folder {root_dir} does not exist: creating the folder and downloading the dataset")

    check_variable_value(name, ["CIFAR-10", "CIFAR-100"])

    if name == 'CIFAR-10':
      # Access to the data (PIL Image, Num_Class)
      train_set = CIFAR10(root = root_dir, train=True, download=True)
    
    # Custom transform-pipeline for data augmentation
    self.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  def __len__(self):
        return len(self.train_set)

  def __getitem__(self, index):
     return 
    

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

# Function to call the Dataset classes
def main():
  dataset = ImagesDataset()


if __name__ == '__main__':
  main()