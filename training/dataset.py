import glob
import os

import cv2
import torch
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    """Dataset class for Chest X-Ray Pneumonia detection.

    Expects directory structure:
        path/
            NORMAL/    *.jpeg  or  *.jpg
            PNEUMONIA/ *.jpeg  or  *.jpg

    collate_fn returns CPU tensors.
    The training loop calls .to(device) for GPU transfer.
    """

    def __init__(self, path: str, transforms=None, limit: int = -1):
        self.path = path
        from random import shuffle, seed
        seed(10)

        # Support both .jpeg and .jpg extensions (Windows-safe path join)
        self.X = (glob.glob(os.path.join(path, '*', '*.jpeg')) +
                  glob.glob(os.path.join(path, '*', '*.jpg')))
        shuffle(self.X)

        if limit > 0:
            print(f"Limiting dataset to {limit} samples")
            self.X = self.X[:limit]

        # Label: 1 = PNEUMONIA, 0 = NORMAL
        self.Y = [
            int(os.path.basename(os.path.dirname(x)) == 'PNEUMONIA')
            for x in self.X
        ]
        self.transforms = transforms
        print(f"Loaded {len(self.X)} samples from {path}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        img   = cv2.imread(self.X[ix])   # HxWxC uint8 BGR
        label = self.Y[ix]
        return img, label

    def collate_fn(self, batch):
        imgs_raw, labels_raw = list(zip(*batch))

        if self.transforms is None:
            raise ValueError("transforms must be provided to collate_fn")

        # Apply transforms per image on CPU (resize, normalize, ToTensor)
        imgs   = torch.cat([self.transforms(img).unsqueeze(0) for img in imgs_raw], dim=0).float()
        labels = torch.cat([torch.tensor(lbl).unsqueeze(0)    for lbl in labels_raw]).float()

        # Returns CPU tensors intentionally.
        # Training loop calls .to(device) for GPU placement.
        return imgs, labels
