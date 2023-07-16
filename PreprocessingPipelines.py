import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os 
import concurrent.futures

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

def augment_image(row, transform_pipeline):
  """
  :param image: PIL Image to modify
  :param transform_pipeline: Pipeline to apply to the image
  :return: Tuple -> (Modified image as a PIL image and the label)
  """
  processed_image = transform_pipeline(row[0])
  return (processed_image, row[1])

# Get the train loader fromt the custom class
def get_train_loader():

  dataset = ImagesDataset()
  print(dataset[0])
  
  return 

class ImagesDataset(Dataset):
  """ Class to work with CIFAR-10 (train split)
  """

  def __init__(self, name='CIFAR-10', root_dir='Datasets', data_augmentation=True, transform = None, train_shape=(64, 32)):
    """ Custom class for dataset loading (download from Torchvision)

    :param name: Name (str) of the dataset to donwload
    :param root_dir: Path (str) to use when saving the dataset
    """

    if os.path.exists(root_dir):
      print("Downloading the dataset in the folder")
    else:
      print(f"Folder {root_dir} does not exist: creating the folder and downloading the dataset")

    # If not checked on the parsing arguments
    check_variable_value(name, ["CIFAR-10", "CIFAR-100"])

    if name == 'CIFAR-10':
      # Access to the data (PIL Image, Num_Class)
      self.train_set = CIFAR10(root = root_dir, train=True, download=True)

    # Data augmentation pipeline to agument the dataset
    if data_augmentation == True:
      self.augment_dataset()
      
      # reload from disk after saving the new images

    if transform != None:
      # Just transform to a tensor and resize for training
      self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Resize(train_shape)])

  def __len__(self):
    return len(self.train_set)

  def __getitem__(self, index):
    image, label = self.train_set[index]
    return self.transform(image), label

  def augment_dataset(self, save_on_disk = True):

    # Composed transformations to augment the dataset
    pipeline = torch.nn.Sequential(
    transforms.RandomCrop(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=20),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToPILImage() # Final conversion to save it on the disk
    )

    # Parallelized data augmentation
    with concurrent.futures.ThreadPoolExecutor() as executor:
      # Submit the processing tasks to the executor
      futures = [executor.submit(augment_image, row, pipeline) for row in self.train_set]
      # Retrieve the results as they become available
      processed_images = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Save augmented images
    torch.save(processed_images, './train.pt')
    return 

def main():
  # Main loop to retrieve the Data loader
  train_loader = get_train_loader()

if __name__ == '__main__':
  main()