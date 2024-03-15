"""
- Creates new files for the training, test, and validation sets on the data
- Trains boosted decision tree on the test and reports the performance on the validation set
"""
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_WIDTH = 322
IMAGE_HEIGHT = 245
SUB_GRID_WIDTH = 12
SUB_GRID_HEIGHT = 9
GRID_BOX_WIDTH = IMAGE_WIDTH // (SUB_GRID_WIDTH - 1)
GRID_BOX_HEIGHT = IMAGE_HEIGHT // (SUB_GRID_HEIGHT - 1)


def save_train_val_test_no_encoding(file_path: str):
    """

    :param file_path:
    :return:
    """


def create_train_val_test_sets(file_path: str):
    """Create the training, validation and testing sets. We have 20 years of data (2001-2021). We will use the last two
    years as our test set (10%), and the next most recent 2 years as the validation set (10%). The remaining 80% of the
    data will be used for training. We also move the "group" field to be the last column for readability.

    Parameters
    ----------
    file_path:

    Returns
    -------
    """
    column_names = ["year", "pixel_position_encoding", "crbn_dioxide", "methane", "ntrs_oxide", "ntrs_oxide", "group"]
    with open("cos_train_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
    with open("cos_validation_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
    with open("cos_test_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

    with open(file_path, 'r') as file:
        file.readline()
        for line in file.readlines():
            row = line.strip().split(",")
            group = row.pop(1)
            row.append(group)
            year = int(row[0])

            if year >= 2020:
                with open("cos_test_data.csv", "a") as f:
                    csv.writer(f).writerow(row)
            elif year >= 2018:
                with open("cos_validation_data.csv", "a") as f:
                    csv.writer(f).writerow(row)
            else:
                with open("cos_train_data.csv", "a") as f:
                    csv.writer(f).writerow(row)


def get_grid_number_from_pixel_id(pixel_id: int) -> int:
    """Return the id of the sub-grid that contains the given pixel

    Parameters
    ----------
    pixel_id

    Returns
    -------
    The id of the sub-grid that contains the given pixel
    """
    true_row = pixel_id // IMAGE_WIDTH
    true_col = pixel_id % IMAGE_WIDTH
    return true_col // GRID_BOX_WIDTH + (true_row // GRID_BOX_HEIGHT) * SUB_GRID_WIDTH


def save_sub_grid_pixel_encoding_test(file_path: str):
    """Create new files that encode the data at the given path with the sub-grid encoding.

    Parameters
    ----------
    file_path

    Returns
    -------
    """
    # Write the column names to the files
    column_names = ["year", "absolute_pixel_position"]
    for i in range(SUB_GRID_WIDTH * SUB_GRID_HEIGHT):
        column_names.append("sub_grid_" + str(i))
    column_names.extend(["crbn_dioxide", "methane", "ntrs_oxide", "ntrs_oxide", "group"])
    with open("conv_train_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
    with open("conv_validation_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
    with open("conv_test_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

    with open(file_path, "r") as file:
        file.readline()

        for line in file.readlines():
            position_encoding = [0] * (SUB_GRID_WIDTH * SUB_GRID_HEIGHT)
            row = line.strip().split(",")
            year = int(row[0])
            pixel_position = int(row[2])
            group = row.pop(1)
            row.append(group)

            position_encoding[get_grid_number_from_pixel_id(pixel_position)] = 1
            encoded_row = row[0:1] + position_encoding + row[2:]

            if year >= 2020:
                with open("conv_test_data.csv", "a") as f:
                    csv.writer(f).writerow(encoded_row)
            elif year >= 2018:
                with open("conv_validation_data.csv", "a") as f:
                    csv.writer(f).writerow(encoded_row)
            else:
                with open("conv_train_data.csv", "a") as f:
                    csv.writer(f).writerow(encoded_row)


def save_sub_grid_pixel_encoding(file_path: str, sub_grid_width: int, sub_grid_height: int):
    """Instead of using on-hot vectors, use integers to represent the "box" and train the model by marking that column
    as "categorical"

    :param file_path:
    :param sub_grid_width:
    :param sub_grid_height:
    :return:
    """
    grid_box_width = IMAGE_WIDTH // (sub_grid_width - 1)
    grid_box_height = IMAGE_HEIGHT // (sub_grid_height - 1)

    def get_grid_number_from_pixel_id_(pixel_position: int):
        true_row = pixel_position // IMAGE_WIDTH
        true_col = pixel_position % IMAGE_WIDTH
        return true_col // grid_box_width + (true_row // grid_box_height) * sub_grid_width

    convert_encoding = np.vectorize(get_grid_number_from_pixel_id_)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    data = data[:, [0, 2, 3, 4, 5, 6, 1]]  # move group number to end of the row
    data[:, 1] = convert_encoding(data[:, 1])

    test_data = 2021 <= data[:, 0]
    data_2019 = data[:, 0] == 2019
    data_2018 = data[:, 0] == 2018
    validation_data = np.logical_or(data_2018, data_2019)
    train_data = data[:, 0] < 2018

    file_name_prefix = str(sub_grid_width) + "x" + str(sub_grid_height)
    column_names = ["year","pixel_position_encoding","crbn_dioxide","methane","ntrs_oxide","srfce_tmp", "group"]
    df = pd.DataFrame(data[test_data])
    df.columns = column_names
    df.to_csv(file_name_prefix + "_conv_test_data.csv", header=True, index=False)
    df = pd.DataFrame(data[validation_data])
    df.columns = column_names
    df.to_csv(file_name_prefix + "_conv_validation_data.csv", header=True, index=False)
    df = pd.DataFrame(data[train_data])
    df.columns = column_names
    df.to_csv(file_name_prefix + "_conv_train_data.csv", header=True, index=False)


def read_features_and_labels(file_path: str, cosine_encoding=True) -> tuple:
    """Open the file at the given path and return two numpy arrays.

    Parameters
    ----------
    file_path
    cosine_encoding

    Returns
    -------
    A tuple containing the feature vector and the labels
    """
    if cosine_encoding:
        inputs = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=range(6))
        labels = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=[6])
    else:
        inputs = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=[0] + [i for i in range(2, 113)])
        labels = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=[113])
    return inputs, labels


def train_and_validate_tree(train_path: str, val_path: str, file_name: str, cosine_encoding=True):
    """Train the decision tree and evaluate its performance on the validation set.
    Parameters
    ----------
    train_path
    val_path

    Returns
    -------

    """
    x_train, y_train = read_features_and_labels(train_path, cosine_encoding)
    x_val, y_val = read_features_and_labels(val_path, cosine_encoding)
    clf = HistGradientBoostingClassifier().fit(x_train, y_train)

    # Save the clf
    # save_file_name = "cos_base_clf.joblib" if cosine_encoding else "conv_base_clf.joblib"
    dump(clf, file_name + "joblib")

    predictions = clf.predict(x_val)
    print(classification_report(y_val, predictions))
    print(confusion_matrix(y_val, predictions))


def load_existing_model(model_path: str, data_path: str, cosine_encoding=True):
    """Load the existing model and save the predictions to a file

    :param model_path:
    :param data_path:
    :param cosine_encoding:
    :return:
    """
    x_val, y_val = read_features_and_labels(data_path, cosine_encoding)
    # Only load data for one year
    rows_to_keep = x_val[:, 0] == 2018
    x_val = x_val[rows_to_keep]
    y_val = y_val[rows_to_keep]

    clf = load(model_path)
    predictions = clf.predict(x_val)
    print(classification_report(y_val, predictions))
    print(confusion_matrix(y_val, predictions))

    # Save predictions
    column_names = ["year","group","pixel_position_encoding","crbn_dioxide","methane","ntrs_oxide","srfce_tmp"]
    df = pd.DataFrame(np.insert(x_val, 1, predictions, axis=1))
    df.columns = column_names
    df.to_csv("./predictions.csv", header=True, index=False)


def train_different_grid_sizes(file_path: str):
    """Create and save datasets with different grid sizes. Train and save models with the new grid sizes and print the
    accuracies.

    :param file_path:
    :return:
    """
    width_height_proportion = (4, 3)

    for i in range(4):
        width = width_height_proportion[0] * (i + 1)
        height = width_height_proportion[1] * (i + 1)
        file_name = str(width) + "x" + str(height) + "_conv"
        save_sub_grid_pixel_encoding(file_path, width, height)
        # train_and_validate_tree(file_name + "_train_data.csv", file_name + "_validation_data.csv", file_name)

        x_train, y_train = read_features_and_labels(file_name + "_train_data.csv")
        x_val, y_val = read_features_and_labels(file_name + "_validation_data.csv")
        clf = HistGradientBoostingClassifier(categorical_features=[1], max_bins=width*height).fit(x_train, y_train)

        # Save the clf
        dump(clf, file_name + ".joblib")

        predictions = clf.predict(x_val)
        print(classification_report(y_val, predictions))
        print(confusion_matrix(y_val, predictions))


def train_given_grid_size(file_path: str, grid_width: int, grid_height: int):
    """Train a grid with the given width and height.

    :param file_path:
    :param grid_width:
    :param grid_height:
    :return:
    """
    file_name = str(grid_width) + "x" + str(grid_height) + "_conv"
    save_sub_grid_pixel_encoding(file_path, grid_width, grid_height)
    # train_and_validate_tree(file_name + "_train_data.csv", file_name + "_validation_data.csv", file_name)

    x_train, y_train = read_features_and_labels(file_name + "_train_data.csv")
    x_val, y_val = read_features_and_labels(file_name + "_validation_data.csv")
    clf = HistGradientBoostingClassifier(categorical_features=[1],
                                         max_bins=255).fit(x_train, y_train)

    # Save the clf
    dump(clf, file_name + ".joblib")

    predictions = clf.predict(x_val)
    print(classification_report(y_val, predictions))
    print(confusion_matrix(y_val, predictions))


def save_base_cos_model(file_prefix: str):
    """

    :param file_name:
    :return:
    """
    x_train, y_train = read_features_and_labels(file_prefix + "_train_data.csv")
    x_val, y_val = read_features_and_labels(file_prefix + "_validation_data.csv")
    clf = HistGradientBoostingClassifier().fit(x_train, y_train)

    # Save the clf
    dump(clf, file_prefix + ".joblib")

    predictions = clf.predict(x_val)
    print(classification_report(y_val, predictions))
    print(confusion_matrix(y_val, predictions))


def tune_hyperparameters(file_prefix: str):
    """Fit models with the varying hyperparameters and plot their accuracies. Return a tuples containing the values of
    the hyperparameters that maximized accuracy, and the accuracy of the model given those hyperparameters. The tuple
    will have format (learning_rate, l2_reg_factor, accuracy)

    Save the optimal model.

    :param file_prefix:
    :return:
    """
    learning_rates = [0.001, 0.01, 0.1, 0.2]
    l2_reg_factors = [0, 0.01, 1, 2, 4, 8]
    x_train, y_train = read_features_and_labels(file_prefix + "_train_data.csv")
    x_val, y_val = read_features_and_labels(file_prefix + "_validation_data.csv")

    # Start by loading the "base" model
    clf = load(file_prefix + ".joblib")
    predictions = clf.predict(x_val)
    best_hyperparameters = 0.1, 0, accuracy_score(predictions, y_val)
    iteration = 0
    _, axis = plt.subplots(1, 6, figsize=(25, 4))
    for l2_reg_factor in l2_reg_factors:
        accuracies = []
        for learning_rate in learning_rates:
            print("Training with l2 = " + str(l2_reg_factor) + ", lr = " + str(learning_rate))
            # Train model with current parameters
            clf = (HistGradientBoostingClassifier(learning_rate=learning_rate, l2_regularization=l2_reg_factor)
                   .fit(x_train, y_train))
            predictions = clf.predict(x_val)

            # Get accuracy and update best hyperparameter
            accuracy = accuracy_score(y_val, predictions)
            accuracies.append(accuracy)
            if accuracy > best_hyperparameters[2]:
                best_hyperparameters = learning_rate, l2_reg_factor, accuracy

        # Plot the results
        plot = axis[iteration]
        plot.plot(learning_rates, accuracies)
        plot.plot(best_hyperparameters[0],
                  best_hyperparameters[2],
                  'g*',
                  label="Highest Accuracy = " + str(round(best_hyperparameters[2], 3)))
        plot.set_title("Accuracy vs. Learning Rate for L2 factor = " + str(l2_reg_factor))
        plot.legend()
        iteration += 1

    plt.savefig(file_prefix + "_tree_hyperparameter_experiments.png")

    # Save model with best hyperparameters
    lr, l2, acc = best_hyperparameters
    clf = HistGradientBoostingClassifier(learning_rate=lr, l2_regularization=l2).fit(x_train, y_train)
    dump(clf, file_prefix + "_optimal_hyperparams.joblib")
    return best_hyperparameters


if __name__ == "__main__":
    # create_train_val_test_sets("cos_k_4_data.csv")
    # save_sub_grid_pixel_encoding("k_4_data.csv")
    # train_and_validate_tree("cos_train_data.csv", "cos_validation_data.csv", cosine_encoding=True)
    # train_and_validate_tree("conv_train_data.csv", "conv_validation_data.csv", cosine_encoding=False)

    # load_existing_model("cos_base_clf.joblib", "cos_validation_data.csv")
    # train_different_grid_sizes("k_4_data.csv")
    # train_given_grid_size("k_4_data.csv", 18, 14)

    # Experiments: try different learning rates and l_2 regularization factors
    # Use the cosing encoding and the 18x14 grid encoding since it had the highest accuracy
    tune_hyperparameters("cos")
    tune_hyperparameters("18x14_conv")
