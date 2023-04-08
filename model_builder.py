import torch
from torch import nn 
import torchvision
from torchinfo import summary

class EfficientNet50:
    def model():
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
        model = torchvision.models.efficientnet_b0(weights=weights).to("cpu")
        for param in model.features.parameters():
            param.requires_grad = False
        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        output_shape = 3
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1280, 
                            out_features=output_shape, # same number of output units as our number of classes
                            bias=True)).to("cpu")
        return model
    
    def info():
        summary(model=EfficientNet50().model(), 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

