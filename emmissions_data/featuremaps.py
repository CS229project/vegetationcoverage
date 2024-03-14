# import util
import numpy as np
import matplotlib  # .pyplot as plt

np.seterr(all='raise')

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None, lambd=1):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.lambd = lambd
        self.x_train = None
        self.y_train = None

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.x_train = X
        self.y_train = y
        self.theta = np.linalg.solve(
            np.matmul(np.matrix.transpose(self.x_train), self.x_train) + self.lambd,
            np.dot(np.matrix.transpose(self.x_train), self.y_train)
        )
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x = X[:, 1]
        poly = np.array(np.power(x, 0))
        for i in range(1, k + 1):
            poly = np.column_stack((poly, np.array(np.power(x, i))))
        return poly
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        poly = self.create_poly(k, X)
        return np.column_stack((poly, np.sin(X[:, 1])))
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.matmul(X, self.theta)
        # *** END CODE HERE ***


# def run_exp(train_path, eval_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
#     train_x, train_y = util.load_dataset(train_path, add_intercept=True)
#     eval_x, eval_y = util.load_dataset(eval_path, add_intercept=True)
#     plt.figure()
#     plt.scatter(train_x[:, 1], train_y)
#
#     for k in ks:
#         '''
#         Our objective is to train models and perform predictions on plot_x data
#         '''
#         # *** START CODE HERE ***
#         lm = LinearModel()
#         if not sine:
#             lm.fit(lm.create_poly(k, train_x), train_y)
#             plot_y = lm.predict(lm.create_poly(k, eval_x))
#         else:
#             lm.fit(lm.create_sin(k, train_x), train_y)
#             plot_y = lm.predict(lm.create_sin(k, eval_x))
#         # *** END CODE HERE ***
#         '''
#         Here plot_y are the predictions of the linear model on the plot_x data
#         '''
#         plt.plot(eval_x[:, 1], plot_y, label='k=%d' % k)
#
#     plt.legend()
#     plt.savefig(filename)
#     plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Degree-3 polynomial regression
    run_exp(train_path, eval_path, ks=[3], filename="degree-3plot.png")

    # Degree-k and sin polynomial regression
    run_exp(train_path, eval_path, filename="polyplot.png")
    run_exp(train_path, eval_path, sine=True, filename="sinplot.png")
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='../clean-data/interval-split/crbn-dioxide-train.csv',
         small_path='small.csv',
         eval_path='../clean-data/interval-split/crbn-dioxide-val.csv')
