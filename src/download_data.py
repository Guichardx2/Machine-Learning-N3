import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
# Define the destination directory
destination_dir = './data'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Move the downloaded files to the destination directory
for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.move(full_file_name, destination_dir)

print(f"Files moved to {destination_dir}")
