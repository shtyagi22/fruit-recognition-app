# download_dataset.py
import zipfile
import requests
import os

url = "https://github.com/Horea94/Fruit-Images-Dataset/archive/master.zip"
output = "fruits.zip"

if not os.path.exists("fruits"):
    print("Downloading dataset...")
    r = requests.get(url)
    with open(output, "wb") as f:
        f.write(r.content)

    print("Extracting dataset...")
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("Fruit-Images-Dataset-master", "fruits")
    os.remove(output)
else:
    print("Dataset already downloaded.")
