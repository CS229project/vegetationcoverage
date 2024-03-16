"""
The code used to choose the most appropriate training/test/validation split of the clean datasets.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_complete_data(source_file_path: str):
    """Plot the data at the given file path

    :param source_file_path:
    :return:
    """
    # Read the data
    complete_data = pd.read_csv(source_file_path)
    complete_data.columns = ["Year", "Carbon Dioxide"]
    complete_data["Year"] = pd.to_datetime(complete_data["Year"], format="%Y")
    complete_data["Carbon Dioxide"] = pd.to_numeric(complete_data["Carbon Dioxide"])

    # Set the year as the index
    complete_data = complete_data.set_index("Year")

    # Plot the data
    plt.figure()
    plt.plot(complete_data)
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions in [Unit]')
    plt.savefig("brazil-emissions.png")


def plot_most_recent_split(source_files: dict, plot=None):
    """Plot the split of the data.

    :param source_files: a list of paths for the training, validation, and test sets
    :return:
    """
    plt.figure(figsize=(4, 4))
    for label, source_file_path in source_files.items():
        data = pd.read_csv(source_file_path)
        data.columns = ["Year", "Carbon Dioxide"]
        data["Year"] = pd.to_datetime(data["Year"], format="%Y")
        data["Carbon Dioxide"] = pd.to_numeric(data["Carbon Dioxide"])

        # Set the year as the index
        complete_data = data.set_index("Year")

        # Plot the data
        plt.plot(complete_data, label=label)

    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions in Tonnes')
    plt.title("Visualize Split for CO2 Data")
    plt.legend()
    plt.savefig("brazil-emissions-by-recent-split.png")


def plot_interval_split(source_files: dict, plot=None):
    """Plot the split of the data.

    :param source_files: a list of paths for the training, validation, and test sets
    :return:
    """
    # plt.figure()
    for label, source_file_path in source_files.items():
        data = pd.read_csv(source_file_path)
        data.columns = ["Year", "Carbon Dioxide"]
        data["Year"] = pd.to_datetime(data["Year"], format="%Y")
        data["Carbon Dioxide"] = pd.to_numeric(data["Carbon Dioxide"])

        # Set the year as the index
        complete_data = data.set_index("Year")

        # Plot the training data as a line
        if label == "train":
            plot.plot(complete_data, label=label)
        # Plot the rest as points
        elif label == "validation":
            plot.plot(complete_data, "bo", label=label)
        else:
            plot.plot(complete_data, "go", label=label)

    plot.xlabel('Year')
    plot.ylabel('CO2 Emissions in [Unit]')
    plot.set_title("Split at Equidistant Intervals")
    plot.legend()
    # plot.savefig("brazil-emissions-by-interval-split.png")


if __name__ == "__main__":
    figure, axis = plt.subplots(1, 2, figsize=(12, 4))
    # plot_complete_data("./clean-data/crbn-dioxide-complete.csv")
    crbn_dioxide_splits = {
        "train": "./clean-data/most-recent-split/crbn-dioxide-train.csv",
        "validation": "./clean-data/most-recent-split/crbn-dioxide-val.csv",
        "test": "./clean-data/most-recent-split/crbn-dioxide-test.csv"
    }
    plot_most_recent_split(crbn_dioxide_splits)
    # plot_most_recent_split(crbn_dioxide_splits, axis[0])
    crbn_dioxide_splits = {
        "train": "./clean-data/interval-split/crbn-dioxide-train.csv",
        "validation": "./clean-data/interval-split/crbn-dioxide-val.csv",
        "test": "./clean-data/interval-split/crbn-dioxide-test.csv"
    }
    # plot_interval_split(crbn_dioxide_splits, axis[1])
    # plt.savefig("compare-splits.png")
