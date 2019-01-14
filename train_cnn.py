import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.manifold import TSNE
from cnn import CNN
from data_loader import DataLoader
from plot_cnn import PlotterCNN


class TrainCNN:
    def __init__(self, cnn, batch_size=50, learning_rate=0.001, epoch=5):
        self.cnn = cnn
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self, prediction_size=10):
        dataset = DataLoader()
        dataset.get_mnist()
        train_data = dataset.get_train_data()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)
        train_loader = data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        test_data = torchvision.datasets.MNIST(root='../input/mnist/mnist/', train=False)
        test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255.
        test_y = test_data.test_labels[:2000]
        loss_func = nn.CrossEntropyLoss()
        plt.ion()

        # training and testing
        for epoch in range(self.epoch):
            for iteration, (data_value, data_label) in enumerate(train_loader):
                batch_x = Variable(data_value)
                batch_y = Variable(data_label)

                output = self.cnn(batch_x)[0]  # cnn output
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iteration % 100 == 0:
                    test_output, last_layer = self.cnn(test_x)
                    label_prediction = torch.max(test_output, 1)[1].data.squeeze()
                    accuracy = (label_prediction == test_y).sum().item() / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
                    # t-Distributed Stochastic Neighbor Embedding
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plotter = PlotterCNN(low_dim_embs, labels)
                    plotter.plot()
            plt.ioff()
        test_output, _ = self.cnn(test_x[:prediction_size])
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print(pred_y, 'prediction number')
        print(test_y[:prediction_size].numpy(), 'real number')


cnn1 = CNN()
train = TrainCNN(cnn1)
train.train(100)
