import matplotlib.pyplot as plt
from matplotlib import cm


class PlotterCNN:
    def __init__(self, low_weights, labels):
        self.low_weights = low_weights
        self.labels = labels

    def plot(self):
        plt.figure(figsize=(20, 6))
        plt.cla()
        x_axis, y_axis = self.low_weights[:, 0], self.low_weights[:, 1]
        for x, y, s in zip(x_axis, y_axis, self.labels):
            c = cm.rainbow(int(255 * s / 9))
            plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(x_axis.min(), x_axis.max())
        plt.ylim(y_axis.min(), y_axis.max())
        plt.title('Visualize last layer')
        plt.show()
        plt.pause(0.01)
