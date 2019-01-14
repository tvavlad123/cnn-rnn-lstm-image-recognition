import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from data_loader import DataLoader
from plot_rnn import PlotterRNN
from rnn import RNN


class TrainRNN:
    def __init__(self, rnn, batch_size=100, learning_rate=0.01, epoch=1):
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self, prediction_size=1000):
        dataset = DataLoader()
        dataset.get_mnist()
        train_data = dataset.get_train_data()
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        test_data = torchvision.datasets.MNIST(root='../input/mnist/mnist/', train=False,
                                               transform=transforms.ToTensor())
        test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[
                 :2000] / 255.
        test_y = test_data.test_labels.numpy().squeeze()[:2000]
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(self.epoch):
            for step, (x, y) in enumerate(train_loader):
                batch_x = Variable(x.view(-1, 28, 28))
                batch_y = Variable(y)
                output = self.rnn(batch_x)
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    test_output = self.rnn(test_x)
                    label_prediction = torch.max(test_output, 1)[1].data.numpy().squeeze()
                    accuracy = sum(label_prediction == test_y) / float(test_y.size)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

        test_output = self.rnn(test_x[:prediction_size].view(-1, 28, 28))
        label_prediction = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print(label_prediction, 'prediction number')
        print(test_y[:prediction_size], 'real number')
        plotter = PlotterRNN(label_prediction, test_y)
        plotter.plot(prediction_size)


rnn1 = RNN()
train = TrainRNN(rnn1)
train.train()
