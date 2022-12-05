from collections.abc import Sequence
from functools import reduce

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models


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


class AlexNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 pretrained: bool = True):
        super(AlexNet, self).__init__()

        self._pretrained = pretrained
        base_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)

        self.extractor = nn.Sequential(base_model.features, base_model.avgpool)
        self.classifier = base_model.classifier
        self.classifier[-1] = nn.Linear(in_features=4096, out_features=n_classes)

        # custom classifier head
        # self.classifier = torch.nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=4096, out_features=2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=1024, out_features=n_classes),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pretrained:
            # don't finetune the core
            self.extractor.eval()
            with torch.no_grad():
                feats = self.extractor(x).flatten(1)
        else:
            # finetune the core model too
            feats = self.extractor(x).flatten(1)
        return self.classifier(feats)
    
    
class DenseNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 pretrained: bool = True):
        super(DenseNet, self).__init__()

        self._pretrained = pretrained
        densenet_kwargs = {}
        base_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None,
                                        **densenet_kwargs)
        self.extractor = base_model.features
        self.classifier = nn.Linear(in_features=1664, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pretrained:
            # don't finetune the core
            self.extractor.eval()
            with torch.no_grad():
                feats = self.extractor(x)
        else:
            # finetune the core model too
            feats = self.extractor(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class VGGBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(VGGBlock, self).__init__()
        convblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(3//2, 3//2)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            convblock,
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGVariant(nn.Module):
    """
    Model from: https://arxiv.org/abs/1807.08775
    """
    def __init__(self,
                 input_shape: tuple,
                 n_classes: int):
        super(VGGVariant, self).__init__()
        channels = [16, 32, 64, 128, 128]
        in_channels = [input_shape[0]] + list(channels[:-1])
        out_channels = channels

        self.features = nn.Sequential(*[VGGBlock(i, o) for i, o in zip(in_channels, out_channels)])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x).flatten(1)
        return self.classifier(out)



if __name__ == '__main__':
    from torchsummaryX import summary
    model = VGGVariant((3, 128, 128), 8)
    summary(model, torch.zeros(1, 3, 128, 128))
