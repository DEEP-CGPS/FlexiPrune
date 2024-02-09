from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import torch
from torch import nn

class ModelParams():
    def __init__(self, model: nn.Module, model_input: torch.Tensor) -> None:
        """
        Initialize ModelParams object.

        Args:
        model (nn.Module): PyTorch model.
        model_input (torch.Tensor): Input tensor for the model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model_input = model_input.to(device)
        
    def get_flops(self) -> int:
        """
        Get the total number of FLOPs.

        Returns:
        int: Total number of FLOPs.
        """
        flops = FlopCountAnalysis(self.model, self.model_input)
        return flops.total()
    
    def get_summary(self) -> None:
        """
        Print the summary of the model.
        """
        print(summary(self.model, tuple(self.model_input.size())))
        
    def get_times_layer(self, layer_type: str = "Conv2d") -> int:
        """
        Get the count of specific layer types in the model.

        Args:
        layer_type (str): Type of layer to count. Defaults to "Conv2d".

        Returns:
        int: Count of specific layer types in the model.
        """
        count_layers = 0      
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and layer_type == "Conv2d":
                count_layers += 1
            if isinstance(module, nn.Linear) and layer_type == "Linear":
                count_layers += 1
        return count_layers           
        
    def get_all_params(self) -> tuple:
        """
        Get all parameters - FLOPs, Conv2d layers count, and Linear layers count.

        Returns:
        tuple: Tuple containing FLOPs, Conv2d layers count, and Linear layers count.
        """
        flops = self.get_flops() * 2
        self.get_summary()
        conv_layers = self.get_times_layer(layer_type="Conv2d")
        linear_layers = self.get_times_layer(layer_type="Linear")
        
        return flops, conv_layers, linear_layers
