import sys
import torch
import re
import os
import pandas as pd
import torch.nn as nn
from .train import get_dataset,test_epoch
import torch
sys.argv = ['']

def evaluate_models(args,metric:bool = True, pruning_methods:str = "random|weight|SenpisFaster",pruning_distribution:str = "PD1|PD2|PD3|PD4|PD5|UNPRUNED",gpd:str = "20|30|50", custom_split=1):
    """
    Evaluate models based on specified criteria.

    Parameters:
    - args (Namespace): Command-line arguments or parameters containing information about the dataset and evaluation.
    - metric (bool, optional): Flag indicating whether to compute and include evaluation metric. Default is True.
    - pruning_methods (str, optional): Regular expression pattern specifying pruning methods for filtering model paths. Default is "random|weight|SenpisFaster".
    - gpd (str, optional): Global Pruning Distribution. Regular expression pattern specifying pruning bases for filtering model paths. Default is "20|30|50".
    - custom_split (int, optional): Custom split value for dataset. Default is 1.

    Returns:
    - pandas.DataFrame: Dataframe containing model information and optionally, evaluation metric.
    """
    model_directory = f"models/{args.dataset}"
    model_paths = []

    for filename in os.listdir(model_directory):
        if filename.endswith(".pth"):
            model_path = os.path.join(model_directory, filename)
            model_paths.append(model_path)

    df = pd.DataFrame({'model_paths': model_paths})
    df['pruning_type'] = df['model_paths'].apply(lambda x: re.search(fr'({pruning_methods})', x).group() if re.search(fr'({pruning_methods})', x) else None)
    df['model_type'] = df['model_paths'].apply(lambda x: re.search(fr'({pruning_distribution})', x).group())
    df['pr_base'] = df['model_paths'].apply(lambda x: re.search(fr'({gpd})', x).group() if re.search(fr'({gpd})', x) else None)
    df['seed'] = df['model_paths'].apply(lambda x: re.search(r'(?<=SEED_)\d+', x).group() if re.search(r'(?<=SEED_)\d+', x) else None)
    df['finetuned'] = df['model_paths'].apply(lambda x: 'FT' in x)
    df['dataset'] = df['model_paths'].apply(lambda x: re.search(fr'{args.dataset}', x).group())

    if metric:
        _, test_loader, num_classes, _ = get_dataset(args, custom_split=custom_split)

        df['metric'] = 0
        df['metric_used'] = args.eval_metric

        criterion = nn.CrossEntropyLoss()
        for i, model_path in enumerate(model_paths):
            model = torch.load(model_path)
            test_loss, test_acc = test_epoch(model, args.device, test_loader, criterion, args.eval_metric, num_classes)
            if torch.is_tensor(test_acc):
                test_acc = test_acc.item()
            df['metric'].iloc[i] = test_acc
            print(f"{args.eval_metric} of model {model_path}: {test_acc:.3f}")

    return df