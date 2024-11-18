import kagglehub
import shutil
import os

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

destination_dir = './data'

os.makedirs(destination_dir, exist_ok=True)

for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.move(full_file_name, destination_dir)

print(f"Files moved to {destination_dir}")
