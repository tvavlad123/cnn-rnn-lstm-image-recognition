from data_loader_batch import DataLoader
import torch
import torch.nn.functional as f
import numpy as np
from torch.autograd import Variable
from cnn_lstm import CnnLstm


class TrainCNNLSTM:
    def __init__(self):
        self.seed = 1
        self.batch_size = 50
        self.test_batch_size = 1000
        self.epoch = 1
        self.learning_rate = 0.01
        self.step = 100
        self.train_loader = None
        self.test_loader = None
        self.model = CnnLstm()

    def load_data(self):
        data_loader = DataLoader()
        self.train_loader = data_loader.get_train_data(self.batch_size)
        self.test_loader = data_loader.get_test_data(self.test_batch_size)

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for iteration, (data, target) in enumerate(self.train_loader):

            data = np.expand_dims(data, axis=1)
            data = torch.FloatTensor(data)

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = self.model(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if iteration % self.step == 0:
                print('Epoch: {} | train loss: {:.4f}'.format(
                    self.epoch, loss.item()))

    def test(self):
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data = np.expand_dims(data, axis=1)
            data = torch.FloatTensor(data)
            print(target.size)

            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += f.nll_loss(
                output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1].data.squeeze()
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))


train = TrainCNNLSTM()
train.load_data()
train.train()
train.test()
