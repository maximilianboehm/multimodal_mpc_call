import os
import csv

# Initialize counters and lists
count = 0
count_all = 0
valid_folders = []

# Root folder to search in
root_folder = "."

# Loop through folders in the root directory
for folder in os.listdir(root_folder):
    if folder.startswith("scripts"):
        data_path = os.path.join(root_folder, folder, "data")
        # Check if the data folder exists
        if os.path.exists(data_path):
            # Loop through subfolders within the data folder
            for data_folder in os.listdir(data_path):
                data_folder_path = os.path.join(data_path, data_folder)
                # Initialize flags to check for file existence
                syncmap_exists = False
                mp4_exists = False
                # Loop through files in the current data folder
                for f in os.listdir(data_folder_path):
                    file_path = os.path.join(data_folder_path, f)
                    # Check if the file is a syncmap
                    if f == "syncmap.json":
                        print(file_path) 
                        syncmap_exists = True  # Set flag for syncmap existence
                    # Check if the file is an mp4 file
                    elif f.endswith(".mp4"):
                        print(file_path) 
                        mp4_exists = True  # Set flag for mp4 existence
                # If both syncmap and mp4 exist in the folder, append to valid_folders list
                if syncmap_exists and mp4_exists:
                    valid_folders.append(data_folder_path)
                    count += 1 
                count_all += 1 
            print("") 

# Write the list of valid folders to a CSV file
with open("valid_embedding_folders.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(valid_folders) 



                
