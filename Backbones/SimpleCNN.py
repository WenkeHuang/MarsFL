import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SimpleCNN(nn.Module):

    def __init__(self, cfg):
        super(SimpleCNN, self).__init__()
        self.name = 'SimpleCNN'
        self.feats = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=cfg.DATASET.n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, 256)

        # last layer
        self.cls = nn.Linear(256, cfg.DATASET.n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        h = h.squeeze()
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        return h

    def classifier(self, h: torch.Tensor) -> torch.Tensor:
        y = self.cls(h)
        return y

    def forward(self, x):
        # h = self.feats(x)
        # h = h.squeeze()
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        x = self.features(x)

        y = self.cls(x)
        return y

class SimpleCNN_sr(nn.Module):

    def __init__(self, cfg):
        super(SimpleCNN_sr, self).__init__()
        self.name = 'SimpleCNN'
        self.feats = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=cfg.DATASET.n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, 256)

        self.mlp = nn.Linear(256, 512)

        # last layer
        self.cls = nn.Linear(256, cfg.DATASET.n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        h = h.squeeze()
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        return h

    def classifier(self, h: torch.Tensor) -> torch.Tensor:
        y = self.cls(h)
        return y

    def featurize(self, x, num_samples=1, return_dist=False):

        features = self.features(x)
        z_params = self.mlp(features)

        z_mu = z_params[:, :self.cls.in_features]
        z_sigma = F.softplus(z_params[:, self.cls.in_features:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.cls.in_features])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        if self.training:

            x = self.featurize(x, return_dist=False)
            y = self.cls(x)
        else:
            x = self.featurize(x, num_samples=self.num_samples, return_dist=False)
            preds = torch.softmax(self.cls(x), dim=1)
            preds = preds.view([self.num_samples, -1, self.cls.out_features]).mean(0)
            y = torch.log(preds)

        return y
