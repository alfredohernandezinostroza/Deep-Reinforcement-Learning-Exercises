from wrappers import make_env
import torch
from typing import Tuple

class deeepQNetwork(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
        )
        fully_connected_input_size = self.convolution(torch.zeros(1, *input_shape)).size()[-1] 
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=fully_connected_input_size, out_features=512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=512, out_features=n_actions)
        )
        self.net = torch.nn.Sequential(
            self.convolution,
            self.linear,
        )

    def forward(self, x: torch.ByteTensor):
        normalized = x / 255.0
        return self.net(normalized)


if __name__ == "__main__":
    enf = make_env("ALE/Pong-v5")