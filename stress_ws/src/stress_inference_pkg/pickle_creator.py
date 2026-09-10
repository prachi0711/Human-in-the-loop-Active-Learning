import pandas as pd
import os
import pickle

folder_path = 'ros2bag_convert/rosbag2_2026_05_13-13_37_02_0.db'
output_file = 'combined_data.pkl'

# Manually load the specific files into a dictionary format
# Adjust these filenames if they are different in 
bvp_df = pd.read_csv(os.path.join(folder_path, "bvp_raw.csv"))
eda_df = pd.read_csv(os.path.join(folder_path, "eda_raw.csv"))

# Create the dictionary structure the ROS node expects
data_dict = {
    'bvp_raw': bvp_df['data'].values, # edit column name accordingly
    'eda_raw': eda_df['data'].values
}

# Save as pickle
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Successfully formatted data for ROS node into {output_file}")
