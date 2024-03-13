import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_one_hot_vector(number_of_groups, y):
    one_hot_y = np.zeros((y.shape[0], number_of_groups))
    one_hot_y[np.arange(y.shape[0]).astype(int), y.astype(int)] = 1
    
    return one_hot_y


def main(lr, data_path, pred_path, start_year, total_years):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training, test and validation set
    total_set = np.arange(0,total_years) + start_year
    print(total_set)
    train_split = 0.8
    eval_split = 0.1

    train_set=np.random.choice(total_set, size=int(total_years*train_split), replace=False)
    eval_test_set = np.setdiff1d(total_set, train_set)
    eval_set = np.random.choice(eval_test_set, size=int(total_years*eval_split), replace=False)
    test_set = np.setdiff1d(eval_test_set, eval_set)

    #Load data from the csv
    df = pd.read_csv(data_path)
    train_data = df.loc[df['year'].isin(train_set), :].to_numpy()[:,1:]
    eval_data = df.loc[df['year'].isin(eval_set), :].to_numpy()[:,1:]
    test_data = df.loc[df['year'].isin(test_set), :].to_numpy()[:,1:]
    
    number_of_groups = np.unique(train_data[:,0]).shape[0]


    train_y = get_one_hot_vector(number_of_groups, train_data[:, 0])
    train_x = train_data[:, 1:]

    eval_y = get_one_hot_vector(number_of_groups, eval_data[:, 0])
    eval_x = eval_data[:, 1:]

    test_y = get_one_hot_vector(number_of_groups, test_data[:, 0])
    test_x = test_data[:, 1:]


    # *** START CODE HERE ***
    # Fit a Multimodal Regression model
    model = MultimodalRegression(theta_0=np.zeros((train_x.shape[1],1)))

    model.fit(train_x,train_y)
    '''
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = model.predict(x_eval)
    
    # Plot dataset
    plt.figure()
    plt.scatter(y_eval, y_predict)

    # Add labels and save to disk
    plt.xlabel('y_true')
    plt.ylabel('y_predicted')
    plt.savefig("poisson_plot.png")


    # *** END CODE HERE ***
    '''


class MultimodalRegression:
    """Multimodal Regression.

    Example usage:
        > clf = MultimodalRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples, number_of_groups).
        """
        prev_theta = self.theta
        i = 0
        stop = False


        while (not stop or not i==1):
            eta = np.dot(x,self.theta)
            gradient = (x.shape[0]**-1)*np.dot(x.T, (y - np.exp(eta)))

            self.theta = self.theta + self.step_size*gradient
            i += 1

            print(np.linalg.norm(prev_theta - self.theta, 1) )
            if (np.linalg.norm(prev_theta - self.theta, 1) < self.eps):
                stop = True
                exit
            else:
                prev_theta = self.theta

            #loss = (x.shape[0]**-1)*np.sum(, axis=0)
        print(f'Converged after: {i} iterations')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        eta = np.dot(x,self.theta)
        y_hat = np.exp(eta)
        y_hat = y_hat.reshape((y_hat.shape[0],))

        return y_hat
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        data_path='~/Downloads/k_4_data.csv',
        pred_path='../data/predictions.csv',
        start_year=2001,
        total_years=21)
