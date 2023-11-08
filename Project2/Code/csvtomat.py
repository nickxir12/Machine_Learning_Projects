import pandas as pd
import scipy.io as sio

# Load the CSV file using pandas
csv_filename = "../MyData/HTRU_2.csv"

try:
    data = pd.read_csv(csv_filename)

    # Convert the pandas DataFrame to a dictionary
    data_dict = data.to_dict()

    # Specify the output .mat file name
    mat_filename = "output_data.mat"

    # Save the data dictionary to a .mat file
    sio.savemat(mat_filename, data_dict)

    print(
        f"CSV file '{csv_filename}' has been converted to MATLAB .mat file '{mat_filename}'."
    )
except Exception as e:
    print(f"An error occurred: {str(e)}")
