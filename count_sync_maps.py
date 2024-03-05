import os
import csv

count = 0
count_all = 0
root_folder = "."
valid_folders = []

for folder in os.listdir(root_folder):
    if folder.startswith("scripts"):
        data_path = os.path.join(root_folder, folder, "data")
        if os.path.exists(data_path):
            for data_folder in os.listdir(data_path):
                data_folder_path = os.path.join(data_path, data_folder)
                syncmap_exists = False
                mp4_exists = False
                for f in os.listdir(data_folder_path):
                    file_path = os.path.join(data_folder_path, f)
                    if f == "syncmap.json":
                        print(file_path)
                        syncmap_exists = True
                    elif f.endswith(".mp4"):
                        print(file_path)
                        mp4_exists = True
                if syncmap_exists and mp4_exists:
                    valid_folders.append(data_folder_path)
                    count += 1
                count_all += 1
            print("")

with open("valid_embedding_folders.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(valid_folders)

print("")
print(count)
print(count_all)

                
