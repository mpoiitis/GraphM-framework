import pandas as pd
import numpy as np
from tqdm import tqdm
from texttable import Texttable


class TADW(object):
    def __init__(self, A, T, node_names, args):
        """
        Setting up the target matrices and arguments. Weights are initialized.
        :param A: Proximity matrix.
        :param T: Text data
        :param args: Model arguments.
        """
        self.A = A
        self.args = args
        self.T = T
        self.node_names = node_names
        self.init_weights()

    def init_weights(self):
        """
        Initialization of weights and loss container.
        """
        self.W = np.random.uniform(0, 1, (self.args.dimension, self.A.shape[0]))
        self.H = np.random.uniform(0, 1, (self.args.dimension, self.T.shape[0]))
        self.losses = []

    def update_W(self):
        """
        Node emebdding matrix update method.
        """
        pass

    def update_H(self):
        """
        Feature embedding matrix update method.
        """
        pass

    def calculate_loss(self, iteration):
        """
        Loss calculation method.
        """
        pass

    def loss_printer(self):
        """
        Function to print the losses in a nice tabular format.
        """
        t = Texttable()
        t.add_rows([["Iteration", "Main loss", "Regul loss I.", "Regul loss II."]])
        t.add_rows(self.losses)
        print(t.draw())

    def optimize(self):
        """
        Gradient descent updates for a given number of iterations.
        """
        self.calculate_loss(0)
        for i in tqdm(range(1, self.args.iter + 1)):
            self.update_W()
            self.update_H()
            self.calculate_loss(i)
        self.loss_printer()

    def compile_embedding(self, ids):
        """
        Method to create embedding using W, H, T and the node ids.
        """
        pass

    def save_embedding(self):
        """
        Saving the embedding on disk.
        """
        print("\nSaving the embedding.\n")
        columns = ["id"] + ["X_" + str(dim) for dim in range(2 * self.args.dimension)]
        ids = np.array(range(self.A.shape[0])).reshape(-1, 1)
        self.W = self.compile_embedding(ids)
        self.out = pd.DataFrame(self.W, columns=columns)
        self.out.to_csv(self.args.output, index=None)


class DenseTADW(TADW):
    """
    Dense Text Attributed DeepWalk Class
    """

    def update_W(self):
        """
        A single update of the node embedding matrix.
        """
        H_T = np.dot(self.H, self.T)
        grad = self.args.lambd * self.W - np.dot(H_T, self.A - np.dot(np.transpose(H_T), self.W))
        self.W = self.W - self.args.alpha * grad
        self.W[self.W < self.args.lower_control] = self.args.lower_control

    def update_H(self):
        """
        A single update of the feature basis matrix.
        """
        inside = self.A - np.dot(np.dot(np.transpose(self.W), self.H), self.T)
        grad = self.args.lambd * self.H - np.dot(np.dot(self.W, inside), np.transpose(self.T))
        self.H = self.H - self.args.alpha * grad
        self.H[self.H < self.args.lower_control] = self.args.lower_control

    def calculate_loss(self, iteration):
        """
        Calculating the losses in a given iteration.
        :param iteration: Iteration round number.
        """
        main_loss = np.sum(np.square(self.A - np.dot(np.dot(np.transpose(self.W), self.H), self.T)))
        regul_1 = self.args.lambd * np.sum(np.square(self.W))
        regul_2 = self.args.lambd * np.sum(np.square(self.H))
        self.losses.append([iteration, main_loss, regul_1, regul_2])

    def compile_embedding(self, ids):
        """
        Saving the embedding on disk.
        """
        to_concat = [ids, np.transpose(self.W), np.transpose(np.dot(self.H, self.T))]
        return np.concatenate(to_concat, axis=1)


class SparseTADW(TADW):
    """
    Sparse Text Attributed DeepWalk Class
    """

    def update_W(self):
        """
        A single update of the node embedding matrix.
        """
        H_T = self.T.transpose().dot(self.H.transpose()).transpose()
        grad = self.args.lambd * self.W - np.dot(H_T, self.A - np.dot(np.transpose(H_T), self.W))
        self.W = self.W - self.args.alpha * grad
        self.W[self.W < self.args.lower_control] = self.args.lower_control

    def update_H(self):
        """
        A single update of the feature basis matrix.
        """
        inside = self.A - self.T.transpose().dot(np.transpose(self.W).dot(self.H).transpose())
        right = self.T.dot(np.dot(self.W, inside).transpose()).transpose()
        grad = self.args.lambd * self.H - right
        self.H = self.H - self.args.alpha * grad
        self.H[self.H < self.args.lower_control] = self.args.lower_control

    def calculate_loss(self, iteration):
        """
        Calculating the losses in a given iteration.
        :param iteration: Iteration round number.
        """
        inside = self.A - self.T.transpose().dot(np.transpose(self.W).dot(self.H).transpose())
        main_loss = np.sum(np.square(inside))
        regul_1 = self.args.lambd * np.sum(np.square(self.W))
        regul_2 = self.args.lambd * np.sum(np.square(self.H))
        self.losses.append([iteration, main_loss, regul_1, regul_2])

    def compile_embedding(self, ids):
        """
        Saving the embedding on disk.
        """
        to_concat = [ids, np.transpose(self.W), self.T.transpose().dot(self.H.transpose())]
        return np.concatenate(to_concat, axis=1)

