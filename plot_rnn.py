import matplotlib.pyplot as plt


class PlotterRNN:
    def __init__(self, prediction, test):
        self.prediction = prediction
        self.test = test

    def plot(self, size):
        plt.figure(1, figsize=(20, 8))
        plt.plot(self.prediction, c='green', label='Predicted')
        plt.plot(self.test[:size], c='orange', label='Actual')
        plt.xlabel("Index")
        plt.ylabel("Predicted/Actual Value")
        plt.title("RNN Classification Result Analysis")
        plt.legend(loc='best')
        plt.show()
