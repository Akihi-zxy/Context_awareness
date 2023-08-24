import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class CustomDataset(Dataset):
    def __init__(self, sentences, images, transform=None):
        texts = pd.read_excel(sentences)
        sentences = texts
        dsc = sentences.iloc[:, 1].astype(str).tolist()
        dsc = clip.tokenize(dsc).to(device)
        self.sentences = dsc
        image_names = sentences.iloc[:, 0].astype(str).tolist()
        self.images = []
        for i in image_names:
            image_path = os.path.join(images, i + '.jpg')
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            self.images.append(image)
        self.images = torch.cat(self.images, dim=0).to(device)
        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        image = self.images[idx]
        return sentence, image

def get_data_loader(excel_data, data_dir, batch_size):
    dataset = CustomDataset(excel_data, data_dir, transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


