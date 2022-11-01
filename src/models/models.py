from collections.abc import Sequence
from functools import reduce

import torch
import torch.nn as nn


class SimpleFeedForward(nn.Module):

    def __init__(self,
                 input_shape,
                 n_classes,
                 units,
                 dropout):
        super(SimpleFeedForward, self).__init__()

        in_units = [input_shape[0]] + list(units[:-1])
        out_units = units
        fc = [nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
            nn.Dropout(d)
        ) for i, o, d in zip(in_units, out_units, dropout)]
        fc.append(nn.Linear(out_units[-1], n_classes))

        self.out = nn.Sequential(*fc)

    def forward(self, x):
        return self.out(x)


class SimpleCNN(nn.Module):

    def __init__(self,
                 input_shape,
                 n_classes,
                 channels=(12, 24, 36),
                 kernels=(7, (3, 1), 3),
                 pools=((4, 4), (2, 2), (2, 2)),
                 cnn_dropout=(0.0, 0.0, 0.2),
                 fc_units=(64,),
                 fc_dropout=(0.2,)
                 ):
        super(SimpleCNN, self).__init__()

        in_channels = [input_shape[0]] + list(channels[:-1])
        out_channels = channels
        kernels = [k if isinstance(k, Sequence) else (k, k) for k in kernels]

        self.cnn = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, padding=(k[0]//2, k[1]//2)),
            nn.BatchNorm2d(num_features=o),
            nn.ReLU(),
            nn.MaxPool2d(p),
            nn.Dropout(d)
        ) for i, o, k, p, d in zip(in_channels, out_channels, kernels, pools, cnn_dropout)])

        t = reduce(lambda x, y: x * y, [p[0] for p in pools], 1)
        f = reduce(lambda x, y: x * y, [p[1] for p in pools], 1)
        cnn_features = out_channels[-1] * (input_shape[1] // t) * (input_shape[2] // f)
        in_features = [cnn_features] + list(fc_units[:-1])
        out_features = fc_units
        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
            nn.Dropout(d)
        ) for i, o, d in zip(in_features, out_features, fc_dropout)])

        self.out = nn.Linear(out_features[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for cnn in self.cnn:
            x = cnn(x)
        x = torch.flatten(x, 1)
        for fc in self.fc:
            x = fc(x)

        return self.out(x)