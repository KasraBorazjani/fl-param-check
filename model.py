import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleModel(nn.Module):
    """
    simpleModel class representing a basic neural network model for image classification.

    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer with input size 28*28 and output size 128.
        fc2 (torch.nn.Linear): Second fully connected layer with input size 128 and output size 64.
        fc3 (torch.nn.Linear): Third fully connected layer with input size 64 and output size 10.

    Methods:
        __init__: Initializes the simpleModel object.
        forward: Defines the forward pass of the model.

    Example:
        model = simpleModel()
        output = model(input_data)
    """

    def __init__(self) -> None:
        """
        Initializes the simpleModel object.
        Defines the architecture of the neural network.

        Layers:
        - fc1: First fully connected layer with input size 28*28 and output size 128.
        - fc2: Second fully connected layer with input size 128 and output size 64.
        - fc3: Third fully connected layer with input size 64 and output size 10.
        """

        super(simpleModel, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x