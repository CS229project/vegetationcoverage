"""
A script to format the emmissions dataset from the following site:
https://ourworldindata.org/co2-and-greenhouse-gas-emissions

Save data into a new CSV with the following columns
- One-hot encoding of country name
- Year
- CO2 (tonnes)
- Methane (tonnes)
- Nitrous oxide (tonnes)
- Warming impact (degrees Celcius)

All of the metrics were calculated per country (not per capita)
"""
import csv


def create_dataset(source_files: dict, destination_file: str):
    """Create a dataset as described in the files docstring using the source files. Save the final dataset in the path
    described by the destination file. The countries are encoded as one-hot vectors.

    :param source_files: List of relative paths for the source files
    :param destination_file: Relative path for the destination file.
    :return: The number of columns in the final dataset
    """

    num_countries = 0
    country_names = []
    ghg_names = []
    data = {}
    for ghg_name, file_path in source_files.items():
        ghg_names.append(ghg_name)
        with open(file_path, 'r') as file:
            file.readline()
            for line in file.readlines():
                country, _, year, value = line.strip().split(",")

                if country not in data:
                    country_names.append(country)
                    num_countries += 1
                    data[country] = {}
                if year not in data[country]:
                    data[country][year] = {}
                if ghg_name not in data[country][year]:
                    data[country][year][ghg_name] = ""

                data[country][year][ghg_name] = value

    # Populate new CSV with the data
    header = country_names + ["Year"] + ghg_names
    dataset = [header]
    current_country_index = 0

    for country in country_names:
        for year in data[country]:
            data_point = [0] * num_countries
            data_point[current_country_index] = 1
            data_point.append(year)
            for ghg in ghg_names:
                if ghg not in data[country][year]:
                    data_point.append(None)
                else:
                    data_point.append(data[country][year][ghg])
            dataset.append(data_point)
        current_country_index += 1

    with open(destination_file, "w+") as dataset_csv:
        dataset_writer = csv.writer(dataset_csv, delimiter=",")
        dataset_writer.writerows(dataset)
    print("Saved data set at ./" + destination_file)

    return len(header)


def validate_dataset(file_path: str, num_cols: int):
    """Validate the dataset on the given path by making sure it has the expected number of columns.

    :param file_path:
    :param num_cols:
    :return:
    """
    with open(file_path, 'r') as file:
        for line in file.readlines():
            assert len(line.strip().split(",")) == num_cols


def create_data_for_country(country: str, source_files: dict):
    """Return a data dictionary for the emissions data for the given country.

    :param country:
    :param source_files:
    :return:
    """
    ghg_names = []
    data = {}
    for ghg_name, file_path in source_files.items():
        ghg_names.append(ghg_name)
        data[ghg_name] = {}
        with open(file_path, 'r') as file:
            file.readline()
            for line in file.readlines():
                country_name, _, year, value = line.strip().split(",")

                if country_name == country:
                    if year not in data[ghg_name]:
                        data[ghg_name][year] = ""
                    data[ghg_name][year] = value

    return data


def create_training_test_validation_sets(data: dict, destination_path: str, ghg_friendly_names: dict):
    """This function will save multiple versions of the dataset:
    - At the root of the destination path, it'll save the entire, unsplit dataset (this will allow us to plot the data
    before it is split)
    - at destination_path/most-recent-split/, it will save the dataset with the most recent 10% of years as the test
    set, the next most recent 10% as the validation set, and the rest as the training set.
    - at destination_path/continuous-split/, it will create a 20% test set that has been sampled at a certain interval
    throughout the years available in the data

    :param data:
    :param destination_path:
    :param ghg_friendly_names:
    :return:
    """
    for ghg_name, ghg_data in data.items():
        dataset = [list(data_point) for data_point in ghg_data.items()]
        header = [["Year", ghg_name]]

        # Save entire dataset
        file_path = destination_path + ghg_friendly_names[ghg_name]
        with open(file_path + "_complete.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(dataset)

        num_years = len(dataset)
        test_val_set_size = num_years // 10  # Using 10% for validation, 10% for test

        # Create "most recent split" dataset
        file_path = destination_path + "most_recent_split/" + ghg_friendly_names[ghg_name]
        with open(file_path + "_train.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + dataset[:8 * test_val_set_size])
        with open(file_path + "_val.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + dataset[8 * test_val_set_size: 9 * test_val_set_size])
        with open(file_path + "_test.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + dataset[9 * test_val_set_size:])

        # Create "interval split" dataset
        file_path = destination_path + "interval_split/" + ghg_friendly_names[ghg_name]
        test_set = [dataset[i] for i in range(num_years - 1, 0, -test_val_set_size)][::-1]
        val_set = [dataset[i] for i in range(num_years - test_val_set_size // 2, 0, -test_val_set_size)][::-1]
        used_indices = [i for i in range(num_years - 1, 0, -test_val_set_size)] + \
                       [i for i in range(num_years - test_val_set_size // 2, 0, -test_val_set_size)]
        train_set = []
        for i in range(num_years):
            if i not in used_indices:
                train_set.append(dataset[i])

        with open(file_path + "_train.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + train_set)
        with open(file_path + "_val.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + val_set)
        with open(file_path + "_test.csv", "w+") as dataset_csv:
            dataset_writer = csv.writer(dataset_csv, delimiter=",")
            dataset_writer.writerows(header + test_set)


if __name__ == "__main__":
    source_files = {"Carbon Dioxide": "./pure_data/annual_co2_emissions_per_country.csv",
                    "Methane": "./pure_data/methane_emissions.csv",
                    "Nitrous Oxide": "./pure_data/nitrous_oxide_emissions.csv",
                    "Surface Temp. Affect": "./pure_data/global_warming_fossil.csv"}
    ghg_friendly_names = {
        "Carbon Dioxide": "crbn_dioxide",
        "Methane": "methane",
        "Nitrous Oxide": "ntrs_oxide",
        "Surface Temp. Affect": "srfce_tmp_afft"
    }
    data_set = create_data_for_country("Brazil", source_files)
    create_training_test_validation_sets(data_set, "./clean_data/", ghg_friendly_names)
