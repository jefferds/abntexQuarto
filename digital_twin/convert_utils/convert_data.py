import pandas as pd


def load_sensor_data_to_dataframe(file_path):
    """
    Loads sensor data from a .txt file into a pandas DataFrame.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the file,
                          or None if an error occurs.
    """
    df = pd.read_csv(file_path, sep="\t", skiprows=2)
    df.columns = df.columns.str.strip()

    if "Fuel Flow  (L/hr)" in df.columns:
        df.rename(columns={"Fuel Flow  (L/hr)": "Fuel Flow (L/hr)"}, inplace=True)

    if "Fuel Flow  (L/hr)" in df.columns:
        df.rename(columns={"Fuel Flow  (L/hr)": "Fuel Flow (L/hr)"}, inplace=True)

    return df


def load_sensor_data_to_parquet(file_path):
    """
    Loads sensor data from a .txt file into a pandas DataFrame and saves it as a .parquet file.

    Args:
        file_path (str): The path to the .txt file.
        output_path (str): The path to the output .parquet file.
    """
    df = load_sensor_data_to_dataframe(file_path)

    output_path = file_path.replace(".txt", ".parquet")

    df.to_parquet(output_path)


if __name__ == "__main__":
    file_path = "../../digital_twin/minilab_data/feb_2025_1/Aq_50000.txt"
    load_sensor_data_to_parquet(file_path)
