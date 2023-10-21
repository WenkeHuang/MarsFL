import torch


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, device):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = torch.nn.functional.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
