import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = mobilenet_v2(weights=None)
        self.network.classifier[1] = nn.Linear(self.network.last_channel, num_classes)

    def forward(self, xb):
        return self.network(xb)
