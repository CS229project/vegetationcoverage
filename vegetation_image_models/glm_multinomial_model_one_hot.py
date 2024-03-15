import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_one_hot_vector(number_of_groups, y):
    #print(y.shape)
    #print(number_of_groups)
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
    eval_y = get_one_hot_vector(number_of_groups, eval_data[:, 0])
    test_y = get_one_hot_vector(number_of_groups, test_data[:, 0])

    train_x = train_data[:, 1:]
    print(train_x[:,0])

    #Feature engineering
    fe = 3
    if fe == 0:
        #No feature engineering
        train_x = train_data[:, 0:]
        eval_x = eval_data[:, 0:]
        test_x = test_data[:, 0:]
    elif fe==1:
        #1 add position sine value to data
        train_x = train_data[:, 2:] + np.reshape(train_data[:, 1], (train_data.shape[0],1))
        eval_x = eval_data[:, 2:] + np.reshape(eval_data[:, 1] ,(eval_data.shape[0],1))
        test_x = test_data[:, 2:] + np.reshape(test_data[:, 1],(test_data.shape[0],1))
    elif fe ==2:
        #Make features quadratic with addition from pixel position
        num_cols = train_data.shape[1]
        train_x = train_data[:, 2:] + np.reshape(train_data[:, 1], (train_data.shape[0],1))
        eval_x = eval_data[:, 2:] + np.reshape(eval_data[:, 1] ,(eval_data.shape[0],1))
        test_x = test_data[:, 2:] + np.reshape(test_data[:, 1],(test_data.shape[0],1))
        for i in range(num_cols):
            train_x = np.append(train_x, np.reshape(train_x[:,i]**2, (train_x.shape[0],1)), 1)
            eval_x = np.append(eval_x, np.reshape(eval_x[:,i]**2, (eval_x.shape[0],1)), 1)
            test_x = np.append(test_x, np.reshape(test_x[:,i]**2, (test_x.shape[0],1)), 1)
    elif fe ==3:
        #Use user pixel one hot encoding (Ugh!)
        train_x = train_data[:, 0:]
        num_x_pixels = int(np.max(train_data[:,1:2]))
        num_y_pixels = int(np.max(train_data[:,2:3]))
        print(num_x_pixels, num_y_pixels)
        pixel_x_one_hot = get_one_hot_vector(num_x_pixels,np.squeeze(train_data[:,0:1]))
        print(pixel_x_one_hot.shape)

    exit

    # Fit a Multimodal Regression model
    if os.path.isfile('./glm_theta_init.txt'):
        theta_0 = np.loadtxt('./glm_theta_init.txt')
    else:
        theta_0 = np.zeros((train_x.shape[1],1))

    max_iter = 100000
    
    model = MultimodalRegression(theta_0=theta_0, max_iter=max_iter, use_mini_batch=True)

    train_losses, eval_losses = model.fit(train_x,train_y, eval_x, eval_y)
    
    plt.figure()
    iterations = np.arange(len(train_losses))
    plt.plot(iterations, train_losses, label="train")
    plt.plot(iterations, eval_losses, label="eval")
    plt.xlabel('iteration')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig("loss_plot.png")

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    y_predict = model.predict(test_x)
    print(np.argmax(y_predict, axis=1, keepdims=True))
    print(np.unique(np.argmax(y_predict, axis=1, keepdims=True)))

    #Save Prediction
    predictions = np.argmax(y_predict, axis=1, keepdims=True)
    np.savetxt('../data/k_' + str(number_of_groups) + '_prediction.txt', predictions)
    

class MultimodalRegression:
    """Multimodal Regression.

    Example usage:
        > clf = MultimodalRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-6,
                 theta_0=None, verbose=True, use_mini_batch=False, batch_size=32000):
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
        self.use_mini_batch=use_mini_batch
        self.batch_size=batch_size

    def fit(self, x, y, eval_x, eval_y, epochs=0):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples, number_of_groups).
        """
        prev_theta = self.theta
        losses = []
        eval_losses = []
        i = 0
        stop = False

        while (not stop):
            if (self.use_mini_batch):
                random_batch_index = np.random.choice(x.shape[0], size=self.batch_size, replace=False)
                x = x[random_batch_index, :]
                y = y[random_batch_index, :]

            eta = np.dot(x,self.theta)
            gradient = (x.shape[0]**-1)*np.dot(x.T, (y - np.exp(eta)))

            self.theta = self.theta + self.step_size*gradient #+ 0.001*self.step_size*self.theta
            i += 1

            delta_theta = np.linalg.norm(prev_theta - self.theta, 1)
            if (delta_theta < self.eps or i > self.max_iter):
                stop = True
                exit
            else:
                prev_theta = self.theta

            if i%100 == 0: 
                loss = (y.shape[0]**-1)*np.multiply(y,np.exp(eta)).sum()
                losses.append(loss)
                eval_eta = np.dot(eval_x,self.theta)
                eval_loss = (eval_y.shape[0]**-1)*np.multiply(eval_y,np.exp(eval_eta)).sum()
                eval_losses.append(eval_loss)
                
                print(f'Delta Theta: {delta_theta} Train loss: {loss} Eval Loss: {eval_loss} iteration: {i}')
                #Write the theta so we can initialize it later
                np.savetxt('./glm_theta_init.txt', self.theta)

        print(f'Converged after: {i} iterations')
        # *** END CODE HERE ***
        return losses, eval_losses

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        eta = np.dot(x,self.theta)
        y_hat = np.argmax(np.exp(eta)/np.exp(eta).sum(axis=1, keepdims=True), axis=1, keepdims=True)
        y_hat = np.exp(eta)/np.exp(eta).sum(axis=1, keepdims=True)

        return y_hat

if __name__ == '__main__':
    main(lr=1,
        data_path='~/Downloads/k_4_data_pixel_posn.csv',
        pred_path='../data/predictions.csv',
        start_year=2001,
        total_years=21)
