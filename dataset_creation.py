import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Create DataFrame
train_df = pd.DataFrame(columns=["img_name", "label"])
train_df["img_name"] = os.listdir("./train")
for idx, i in enumerate(os.listdir("./train")):
    if "cat" in i:
        train_df.at[idx, "label"] = 0
    elif "dog" in i:
        train_df.at[idx, "label"] = 1

train_df.to_csv('train_csv.csv', index=False, header=True)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = pd.read_csv(annotation_file)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.image_files.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
