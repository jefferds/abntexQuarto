import pandas as pd


def load_sensor_data_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path, sep="\t", skiprows=2)

        df.columns = df.columns.str.strip()

        if "Fuel Flow  (L/hr)" in df.columns:
            df.rename(columns={"Fuel Flow  (L/hr)": "Fuel Flow (L/hr)"}, inplace=True)

        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(
            f"Error: The file '{file_path}' is empty or contains no data after skipping rows."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


try:

    real_file_df = load_sensor_data_to_dataframe("data.txt")
    if real_file_df is not None:
        print("\nDataFrame loaded successfully from 'data.txt'!")
        print(real_file_df.head())
    else:
        print("\nFailed to load DataFrame from 'data.txt'.")
except IOError:
    print("Could not write to 'data.txt' for the real file example.")
