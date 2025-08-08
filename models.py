import torch
import torch.nn as nn
import torchvision.models as models

class SCCModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=2):
        super(SCCModel, self).__init__()
        self.model = self._get_model(model_name)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_model(self, model_name):
        if model_name == 'densenet121':
            return models.densenet121(pretrained=True)
        elif model_name == 'efficientnet_b4':
            return models.efficientnet_b4(pretrained=True)
        elif model_name == 'inception_v3':
            return models.inception_v3(pretrained=True, aux_logits=False)
        elif model_name == 'vgg19':
            return models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        return self.model(x)