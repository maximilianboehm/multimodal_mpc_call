import os
import csv

count = 0
count_all = 0
root_folder = "."
valid_folders = []
for folder in os.listdir(root_folder):
    if folder.startswith("scripts"):
        for sub_folder in os.listdir(os.path.join(root_folder, folder)):
            if sub_folder == "data":
                for data_folder in os.listdir(os.path.join(root_folder, folder, "data")):
                    syncmap_exists = False
                    mp4_exists = False
                    for f in os.listdir(os.path.join(root_folder, folder, "data", data_folder)):
                        if f == "syncmap.json":
                            print(os.path.join(root_folder, folder, "data", data_folder,f))
                            syncmap_exists = True
                        if f.endswith(".mp4"):
                            print(os.path.join(root_folder, folder, "data", data_folder,f))
                            mp4_exists = True
                    if syncmap_exists and mp4_exists:
                        valid_folders.append(os.path.join(root_folder, folder, "data", data_folder))
                        count += 1
                    count_all += 1
                print("")
                
with open("valid_embedding_folders.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(valid_folders)
    
print("")     
print(count)
print(count_all)
            
                