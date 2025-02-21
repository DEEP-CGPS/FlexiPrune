import numpy as np
from typing import Union

class PruningDistributionCalculator:
    """
    A class to calculate the pruning distribution for different neural network architectures.
    
    Attributes:
        model (str): The model architecture (e.g., 'AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19').
        pruning_distributions (dict): Predefined pruning distributions for each model and pruning distribution type (PD).
    """
    
    def __init__(self, model: str):
        """
        Initializes the PruningDistributionCalculator with a specified model architecture.
        
        Parameters:
            model (str): The model architecture. Must be one of ['AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19'].
        
        Raises:
            ValueError: If the provided model is not supported.
        """
        self.model = model
        self.pruning_distributions = {
            "AlexNet": {
                1: np.array([50, 50, 50, 50, 50]),
                2: np.array([34, 48, 56, 66, 76]),
                3: np.array([70, 60, 44, 30, 18]),
                4: np.array([68, 42, 28, 40, 70]),
                5: np.array([32, 56, 76, 54, 32]),
            },
            "VGG11": {
                1: np.array([50, 50, 50, 50, 50, 50, 50, 50]),
                2: np.array([20, 40, 46, 50, 58, 60, 68, 72]),
                3: np.array([70, 66, 58, 58, 40, 40, 34, 24]),
                4: np.array([20, 40, 60, 62, 62, 60, 18, 18]),
                5: np.array([68, 52, 40, 40, 50, 52, 72, 72]),
            },
            "VGG13": {
                1: np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
                2: np.array([26, 28, 38, 48, 58, 60, 78, 70, 70, 72]),
                3: np.array([72, 70, 64, 60, 58, 50, 38, 28, 18, 10]),
                4: np.array([70, 68, 60, 50, 28, 28, 46, 60, 68, 70]),
                5: np.array([28, 38, 48, 58, 68, 66, 58, 46, 36, 28]),
            },
            "VGG16": {
                1: np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
                2: np.array([20, 40, 46, 46, 50, 50, 50, 60, 60, 60, 68, 70, 70]),
                3: np.array([70, 70, 62, 62, 50, 50, 50, 40, 40, 40, 34, 34, 32]),
                4: np.array([30, 30, 48, 48, 60, 60, 60, 60, 60, 60, 30, 30, 30]),
                5: np.array([66, 64, 50, 50, 40, 40, 40, 50, 50, 50, 64, 66, 66]),
            },
            "VGG19": {
                1: np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
                2: np.array([36, 38, 42, 44, 46, 48, 48, 50, 54, 56, 58, 60, 64, 66, 68, 70]),
                3: np.array([70, 68, 66, 64, 60, 54, 52, 50, 48, 44, 38, 36, 32, 30, 28, 24]),
                4: np.array([70, 68, 58, 50, 48, 44, 38, 40, 44, 46, 48, 52, 58, 62, 66, 70]),
                5: np.array([30, 38, 40, 44, 48, 50, 58, 60, 70, 62, 60, 50, 46, 38, 32, 26]),
            },
        }
        
        if self.model not in self.pruning_distributions:
            raise ValueError("Unsupported model. Choose from ['AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19'].")

    def calculate(self, GPR: Union[int, float], PD: int) -> np.ndarray:
        """
        Calculate the pruning distribution based on the global pruning ratio (GPR) and pruning distribution type (PD).
        
        Parameters:
            GPR (Union[int, float]): Global Pruning Ratio, expected range [0, 100].
            PD (int): Pruning Distribution type, expected range [1, 5].
        
        Returns:
            np.ndarray: An array representing the pruning distribution.
        """
        GPR = min(max(GPR, 0), 100)
        PD = min(max(PD, 1), 5)
        
        PD_new = np.trunc(self.pruning_distributions[self.model][PD] * GPR / 50)
        
        PD_output_size = len(self.pruning_distributions[self.model][PD]) + 2
        PD_output = np.zeros(PD_output_size)
        PD_output[:len(PD_new)] = PD_new
        PD_output[-2] = GPR
        PD_output[-1] = GPR
        
        return PD_output