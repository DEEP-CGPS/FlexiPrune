import sys
import torch
import re
import os
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
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
    df['gpd_base'] = df['model_paths'].apply(lambda x: re.search(fr'({gpd})', x).group() if re.search(fr'({gpd})', x) else None)
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

def box_plot_distribution(df:pd.DataFrame(),list_pruning:list,list_gpd_base:list, plot_name:str = 'PD_BOXPLOT.png', finetuned:bool = True):
    """
    Generate a box plot to visualize distribution of metrics across different model types based on pruning and GPD base.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data to be plotted.
        list_pruning (list): List of pruning types to include in the plot.
        list_gpd_base (list): List of GPD base types to include in the plot.
        plot_name (str, optional): Name of the output plot file. Default is 'PD_BOXPLOT.png'.

    Returns:
        None

    Example:
        box_plot_distribution(df, ['pruning_type_1', 'pruning_type_2'], ['gpd_base_1', 'gpd_base_2'], 'my_plot.png')
    """
    # Filter DataFrame based on pruning types and GPD bases
    df_pruned = df[(df['pruning_type'].isin(list_pruning)) | (df['pruning_type'] != df['pruning_type'])]
    df_pruned = df[(df['gpd_base'].isin(list_gpd_base)) | (df['pruning_type'] != df['pruning_type'])]
    df_pruned = df_pruned[(df_pruned['finetuned'] == finetuned) | (df['pruning_type'] != df['pruning_type'])][['model_type', 'metric']]
    df_pruned = df_pruned.sort_values(by='model_type')

    # Set plot size and define color palette
    plt.rcParams['figure.figsize'] = [10, 5]
    palette = sns.color_palette("husl", len(df_pruned['model_type'].unique()))

    # Create box plot with specified palette
    ax = sns.boxplot(x='model_type', y='metric', data=df_pruned, palette=palette)
    
    # Add stripplot with specified color and attributes
    ax = sns.stripplot(x='model_type', y='metric', data=df_pruned, color="orange", jitter=0.3, size=4)
    
    # Customize grid lines
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
    # Set plot labels
    ax.set(xlabel='Pruning Distributions', ylabel=df.metric_used.unique()[0].upper())
    
    # Save the plot as an image
    plt.savefig(plot_name, dpi=600)
    
    # Show the plot
    plt.show()


def bar_plot_distribution(df: pd.DataFrame, pdis: str, miny: int = 70, maxy: int = 100, file_name: str = 'PD_DISTRIBUTION_EXAMPLE.png', finetuned: bool = True):
    """
    Generate a bar plot to visualize distribution of metrics for a specific model type across different pruning methods.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data to be plotted.
        pdis (str): The specific model type for which the distribution of metrics will be plotted.
        miny (int, optional): Minimum value for the y-axis. Default is 70.
        maxy (int, optional): Maximum value for the y-axis. Default is 100.
        finetuned (bool, optional): Whether to include only fine-tuned models. Default is True.
        file_name (str, optional): Name of the output plot file. Default is 'PD_DISTRIBUTION_EXAMPLE.png'.

    Returns:
        None

    Example:
        bar_plot_distribution(df, 'model_type_1', 70, 100, True, 'my_plot.png')
    """
    # Filter DataFrame based on specified model type and fine-tuned condition
    df_pruned = df[(df['model_type'] == pdis) | (df['pruning_type'] != df['pruning_type'])]
    df_pruned = df_pruned[(df_pruned['finetuned'] == finetuned) | (df_pruned['pruning_type'] != df_pruned['pruning_type'])]
    
    # Fill NaN values with 'Model UNPRUNED'
    df_pruned.fillna('Model UNPRUNED', inplace=True)
    
    # Sort DataFrame by pruning type
    df_pruned = df_pruned.sort_values(by='pruning_type')

    # Set plot size
    plt.rcParams['figure.figsize'] = [10, 5]

    # Create bar plot
    ax = sns.barplot(data=df_pruned, x='pruning_type', y='metric', hue='gpd_base')

    # Set labels and title
    ax.set(xlabel='Pruning Methods', ylabel=df_pruned.metric_used.unique()[0].upper(), title=pdis)
    plt.legend(title='Global PD', loc='lower left')
    plt.ylim(miny, maxy)

    # Add percentage labels to the bars
    for i in ax.containers:
        ax.bar_label(i, fmt="%.1f%%", fontsize=8, padding=3)
    
    # Save the plot as an image
    plt.savefig(file_name, dpi=600)
    
    # Show the plot
    plt.show()

def bar_plot_method(df: pd.DataFrame, pruning_type: str, miny: int = 70, maxy: int = 100, finetuned: bool = True, file_name: str = 'PRUNING_METHOD_EXAMPLE.png'):
    """
    Generate a bar plot to visualize the performance metrics of different model types for a specific pruning method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data to be plotted.
        pruning_type (str): The specific pruning method for which the performance metrics will be plotted.
        miny (int, optional): Minimum value for the y-axis. Default is 70.
        maxy (int, optional): Maximum value for the y-axis. Default is 100.
        finetuned (bool, optional): Whether to include only fine-tuned models. Default is True.
        file_name (str, optional): Name of the output plot file. Default is 'PRUNING_METHOD_EXAMPLE.png'.

    Returns:
        None

    Example:
        bar_plot_method(df, 'pruning_type_1', 70, 100, True, 'my_plot.png')
    """
    # Filter DataFrame based on specified pruning method and fine-tuned condition
    df_pruned = df[(df['pruning_type'] == pruning_type) | (df['pruning_type'] != df['pruning_type'])]
    df_pruned = df_pruned[(df_pruned['finetuned'] == finetuned) | (df_pruned['pruning_type'] != df_pruned['pruning_type'])]
    
    # Replace 'UNPRUNED' model type with 'MODEL UNPRUNED'
    df_pruned.model_type[df_pruned.model_type == 'UNPRUNED'] = 'MODEL UNPRUNED'
    
    # Fill NaN values with 'MODEL UNPRUNED'
    df_pruned.fillna('MODEL UNPRUNED', inplace=True)
    
    # Sort DataFrame by model type
    df_pruned = df_pruned.sort_values(by='model_type')
    
    # Set plot size
    plt.rcParams['figure.figsize'] = [15, 5]
    
    # Create bar plot
    ax = sns.barplot(x='model_type', y='metric', data=df_pruned, hue='gpd_base')

    # Set labels and title
    ax.set(xlabel='Pruning Distributions', ylabel=df_pruned.metric_used.unique()[0].upper(), title=pruning_type.upper())
    plt.legend(title='Global PR', loc='lower left')
    plt.ylim(miny, maxy)
    
    # Add percentage labels to the bars
    for i in ax.containers:
        ax.bar_label(i, fmt="%.1f%%", fontsize=8, padding=3)
    
    # Save the plot as an image
    plt.savefig(file_name, dpi=600)
    
    # Show the plot
    plt.show()