# FlexiPrune

Library for pruning convolutional neural networks by varying the pruning distribution.

## Installation

FlexiPrune can be installed using pip:

```bash
pip install flexiprune==0.1.0
```

or if you want to run the latest version of the code, you can install from git:

```bash
git clone https://github.com/DEEP-CGPS/FlexiPrune
pip3 install -r requirements.txt
```

****

## Usage

### Main function

The main function "flexiprune"  provides all the tools necessary to prune, train, and generate performance metrics by varying the pruning distribution. 

#### Parameters

Parse command-line arguments for configuring and training a neural network model.
    
- `model_architecture (str):` Specify the architecture of the model (e.g., VGG16, AlexNet, etc.).
- `method (str):` Specify the training method (e.g., SenpisFaster, random, weight).
- `dataset (str):` Specify the dataset for training (e.g., CIFAR10, "Name of custom dataset").
- `batch_size (int):` Set the batch size for training.
- `num_epochs (int):` Specify the number of training epochs.
- `learning_rate (float):` Set the learning rate for the optimizer.
- `optimizer_val (str):` Specify the optimizer for training (e.g., SGD, Adam, etc.).
- `model_type (str):` Specify the type of the model (e.g., PRUNED or UNPRUNED).
- `device (str):` Specify the device for training (e.g., "cuda:0" for GPU).
- `model_input (torch.Tensor):` Input tensor for the model (default is a tensor of ones).
- `eval_metric (str):` Specify the evaluation metric (e.g., accuracy, f1).
- `seed (int):` Set the seed for random pruning operations.
- `list_pruning (list):` Specify the list of pruning ratios for each layer.

#### Minimal working example

```python

## 1- Definition of arguments for function usage

import sys
import torch
import torchvision
from flexiprune import *
import argparse
sys.argv = ['']

import argparse
import torch

parser = argparse.ArgumentParser(description='Parameters for training')

parser.add_argument('--model_architecture', type=str, default="VGG16", 
                    help='Specify the architecture of the model (e.g., VGG16, AlexNet, etc.).')

parser.add_argument('--method', type=str, default="random", 
                    help='Specify the training method (e.g., SenpisFaster, random, weight).')

parser.add_argument('--dataset', type=str, default="CIFAR10", 
                    help='Specify the dataset for training (e.g., CIFAR10, "Name of custom dataset").')

parser.add_argument('--batch_size', type=int, default=8, 
                    help='Set the batch size for training.')

parser.add_argument('--num_epochs', type=int, default=1, 
                    help='Specify the number of training epochs.')

parser.add_argument('--learning_rate', type=float, default=1e-3, 
                    help='Set the learning rate for the optimizer.')

parser.add_argument('--optimizer_val', type=str, default="SGD", 
                    help='Specify the optimizer for training (e.g., SGD, Adam, etc.).')

parser.add_argument('--model_type', type=str, default="UNPRUNED", 
                    help='Specify the type of the model (e.g., PRUNED or UNPRUNED).')

parser.add_argument('--device', type=str, default=None, 
                    help='Specify the device for training (e.g., "cuda:0" for GPU).')

parser.add_argument('--model_input', default=torch.ones((1, 3, 224, 224)), 
                    help='Input tensor for the model (default is a tensor of ones).')

parser.add_argument('--eval_metric', default="accuracy", 
                    help='Specify the evaluation metric (e.g., accuracy, f1).')

parser.add_argument('--seed', type=int, default=23, 
                    help='Set the seed for random pruning operations.')

parser.add_argument('--list_pruning', type=list, 
                    default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0], 
                    help='Specify the list of pruning ratios for each layer.')

args = parser.parse_args()


args = parser.parse_args()

if args.device is None:
    import torch
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Get Model, DATASET and TRAIN

model = get_model(10, args)
train_loader, test_loader, num_classes, trainset = get_dataset(args)
train_model(train_loader = train_loader,
            test_loader = test_loader,
            model = model,
            num_classes = num_classes,
            args = args)

#Prune trained model:

model = torch.load(f'models/{args.dataset}/{args.model_architecture}_{args.dataset}_{args.model_type}.pth')
model.to(args.device)
args.model_type = f'your_model_name_with_out_fine_tunning'
prune_model(model,num_classes,trainset, args)

```

### Submodules

flexiprune contains 8 modules that allow to train, prune, generate result tables, and identify model properties (e.g., parameters, number of layers):

#### modelParams:

Allows to obtain the total number of FLOPs, to generate the model summary, to obtain the number of convolutional layers and the FC.

#### train_epoch:

It is optional but it helps to train an epoch of the model, normally it is not used directly but it is used by the train_model module.

#### test_epoch:

It is optional but it helps to perform the test during the epoch of the model, normally it is not used directly but it is used by the train_model module.

#### train_model:

Given the input arguments, allows to train the desired convolutional neural network.

#### get_model:

Returns the desired model.

#### get_dataset:

Returns the desired dataset.

#### prune_model:

Prunes the model, taking into account the arguments.

#### evaluate_models:

Returns a dataframe containing the summary of the pruned model information, this to facilitate its later analysis.

## Citing

If you use this software for research or application purposes, please use the following citation:

```bibtex
@article{ ,
  title = {},
  journal = {SoftwareX},
  volume = {},
  pages = {},
  year = {},
  issn = {},
  doi = {},
  url = {},
  author = {},
}
