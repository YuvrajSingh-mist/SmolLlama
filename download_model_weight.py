import gdown
import os

file_id = "1efKljK-cnM1NuF2Mgx86BljQuEI1B4Em"
url = f"https://drive.google.com/uc?id={file_id}"

os.makedirs('weights/', exist_ok=True)

output = "weights/snapshot.pt"
gdown.download(url, output, quiet=False)

print("Model downloaded successfully!")