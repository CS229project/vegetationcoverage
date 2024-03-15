"""
This file uses feature maps and polynomial regression with L2 regularization.

We perform the experiment multiple times with different permutation of lambda (regularization term) and feature maps.
We evaluate the performance of the models using the mean squared error
"""
from featuremaps import LinearModel
import numpy as np
import matplotlib.pyplot as plt


def add_intercept(x: np.ndarray):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], 2), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1] = x

    return new_x


def load_dataset(csv_path: str, add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x: np.ndarray):
        global add_intercept
        return add_intercept(x)

    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=[0])
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=[1])

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def get_mse(actual, predictions) -> float:
    """Return the MSE for the given data
    :param actual:
    :param predictions:
    :return:
    """
    return np.average(np.square(actual - predictions))


def get_mape(actual, predictions) -> float:
    """Return the MAPE for the given data
    :param actual:
    :param predictions:
    :return:
    """
    return np.average(np.abs(((actual - predictions) / actual)))


def experiment(train_paths: dict, eval_paths: dict, lambds: list, ks: list) -> dict:
    """Return the best values for the hyperparameters for polynominal regression on each dataset.
    Should return a dictionary:
    {"ghg_name": {
        "MSE": (lambda, k, MSE, MAPE),
        "MAPE": (lambda, k, MSE, MAPE)
    }}

    Plot the MSE and MAPE for each k, each value of lambda and each greenhouse gas

    :param train_paths:
    :param eval_paths:
    :param lambds:
    :param ks:
    :return:
    """
    # 18x4
    results = {}
    for ghg, train_path in train_paths.items():
        results[ghg] = {}
        train_x, train_y = load_dataset(train_path, add_intercept=True)
        eval_x, eval_y = load_dataset(eval_paths[ghg], add_intercept=True)
        lowest_mse = (None, None, None, None)
        lowest_mape = (None, None, None, None)

        iteration = 0
        figure, axis = plt.subplots(2, 6, figsize=(18, 8))
        for lambd in lambds:
            mses = []
            mapes = []

            for k in ks:
                # Fit the model
                lm = LinearModel(lambd)
                lm.fit(lm.create_poly(k, train_x), train_y)
                predictions = lm.predict(lm.create_poly(k, eval_x))

                mse = get_mse(eval_y, predictions)
                mses.append(mse)
                mape = get_mape(eval_y, predictions)
                mapes.append(mape)

                if lowest_mse[2] is None or mse < lowest_mse[2]:
                    lowest_mse = lambd, k, mse, mape
                if lowest_mape[3] is None or mape < lowest_mape[3]:
                    lowest_mape = lambd, k, mse, mape

            # Plot the results
            plot = axis[0, iteration]
            plot.plot(ks, mses)
            plot.plot(lowest_mse[1], lowest_mse[2], 'g*', label="Lowest MSE = " + str(round(lowest_mse[2], 3)))
            plot.set_title("MSE vs k for lambda=" + str(lambd))
            plot.legend()
            plot = axis[1, iteration]
            plot.plot(ks, mapes)
            plot.plot(lowest_mape[1], lowest_mape[3], 'g*', label="Lowest MAPE = " + str(round(lowest_mape[3], 3)))
            plot.set_title("MAPE vs k for lambd=" + str(lambd))
            plot.legend()
            plt.savefig(ghg + " Polynominal Regression Experiment Results.png")
            iteration += 1

        results[ghg]["MSE"] = lowest_mse
        results[ghg]["MAPE"] = lowest_mape

    return results


def plot_best_models(train_paths: dict, eval_paths: dict, best_hyperparams: dict):
    """Create a plot of the training data and validation data. Plot the model predictions on top

    :param train_paths:
    :param eval_paths:
    :param best_hyperparams:
    :return:
    """
    figure, axis = plt.subplots(1, 4, figsize=(18, 4))
    iteration = 0
    for ghg, train_path in train_paths.items():
        plot = axis[iteration]
        iteration += 1
        train_x, train_y = load_dataset(train_path, add_intercept=True)
        eval_x, eval_y = load_dataset(eval_paths[ghg], add_intercept=True)

        # Train the ideal model (start with using MAPE)
        lambd, k, _, mape = best_hyperparams[ghg]["MAPE"]
        lm = LinearModel()
        lm.fit(lm.create_poly(k, train_x), train_y)
        predictions = lm.predict(lm.create_poly(k, eval_x))

        plot.plot(train_x[:, 1], train_y, label="training data")
        plot.plot(eval_x[:, 1], eval_y, label="validation set")
        plot.plot(eval_x[:, 1], predictions, label='k=' + str(k) + ', lambda=' + str(lambd))
        plot.set_title(ghg + " Best Fit, MAPE=" + "%.2f" % mape)
        plot.legend()

    plt.savefig("Best Emissions Data Fits.png")


if __name__ == "__main__":
    train_paths = {
        "Carbon Dioxide": 'clean_data/most_recent_split/crbn_dioxide_train.csv',
        "Methane": 'clean_data/most_recent_split/methane_train.csv',
        "Nitrous Oxide": 'clean_data/most_recent_split/ntrs_oxide_train.csv',
        "Surface Temp. Effect": 'clean_data/most_recent_split/srfce_tmp_afft_train.csv'
    }
    eval_paths = {
        "Carbon Dioxide": 'clean_data/most_recent_split/crbn_dioxide_val.csv',
        "Methane": 'clean_data/most_recent_split/methane_val.csv',
        "Nitrous Oxide": 'clean_data/most_recent_split/ntrs_oxide_val.csv',
        "Surface Temp. Effect": 'clean_data/most_recent_split/srfce_tmp_afft_val.csv'
    }
    # lambds = [0.001, 0.01, 0.1, 0.25, 0.5, 1]
    lambds = [1, 2, 3, 4, 5, 6]
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    results = experiment(train_paths, eval_paths, lambds, ks)
    plot_best_models(train_paths, eval_paths, results)
