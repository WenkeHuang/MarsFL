from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch

class AFLLocal(LocalMethod):
    NAME = 'AFLLocal'

    def __init__(self, args, cfg):
        super(AFLLocal, self).__init__(args, cfg)

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        test_loader = kwargs['test_loader']

        loss_dict = kwargs['loss_dict']
        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i])
            _, test_loss = self.local_validate(nets_list[i],test_loader[priloader_list[i].dataset.data_name])
            loss_dict[i] = test_loss

    def train_net(self, index, net, train_loader):
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

    def local_validate(self, model,testDataloader):
        model.eval()
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        batch_loss = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testDataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                loss = criterion(output, labels)
                batch_loss.append(loss.item())
        test_loss = sum(batch_loss) / len(batch_loss)
        # print('| Client id:{} | Test_Loss: {:.3f} | Test_Acc: {:.3f}'.format(self.client_id, test_loss, test_acc))

        return batch_loss, test_loss