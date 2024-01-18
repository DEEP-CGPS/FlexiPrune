from torchvision.transforms import transforms
from typing import List, Tuple
import torch
from torchvision import datasets
from torch.utils.data import random_split, Dataset

def dataset_list(data_dir: str) -> Tuple[List[Dataset], List[Dataset], List[str]]:
    """
    Splits the dataset into train and test sets and retrieves class names.

    Args:
    data_dir (str): Path to the dataset.

    Returns:
    Tuple[List[Dataset], List[Dataset], List[str]]: Tuple containing train, test datasets and class names.
    """
    image_datasets = datasets.ImageFolder(data_dir)
    class_names = image_datasets.classes
    
    proportions = [.80, .20]
    lengths = [int(p * len(image_datasets)) for p in proportions]
    lengths[-1] = len(image_datasets) - sum(lengths[:-1])

    generator = torch.Generator().manual_seed(42)
    train_idx, test_idx = random_split(image_datasets, lengths, generator=generator)
    
    train_list = [image_datasets[i] for i in train_idx.indices]
    test_list = [image_datasets[i] for i in test_idx.indices]
    
    return train_list, test_list, class_names

class CustomDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: transforms.Compose = None) -> None:
        """
        Custom Dataset class.

        Args:
        dataset (Dataset): Dataset object.
        transform (transforms.Compose, optional): Transformations to apply to the dataset.
        """
        self.images = [dataset[x][0] for x in range(len(dataset))]
        self.labels = [dataset[x][1] for x in range(len(dataset))]
        self.transform = transform
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve item from the dataset.

        Args:
        idx (int): Index to retrieve the item.

        Returns:
        Tuple[torch.Tensor, int]: Tuple containing image tensor and label.
        """
        label = self.labels[idx]
        image = self.images[idx]      
        image = self.transform(image)
        return image, label
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        int: Length of the dataset.
        """
        return len(self.labels)
