import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms

class MultimodalFakeNewsDataset(Dataset):
    def __init__(self, csv_file, image_dir, max_len=128):
        self.df = pd.read_csv(csv_file, comment='#')
        self.image_dir = image_dir
        self.max_len = max_len

        self.df['fake_or_real'] = self.df['fake_or_real'].astype(str).str.lower().str.strip()
        self.df['label'] = self.df['fake_or_real'].map({'fake': 1, 'real': 0}).fillna(0).astype(int)

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['news_text'])

        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        img_name = str(row['path']).strip() if pd.notna(row['path']) else ""
        img_path = os.path.join(self.image_dir, img_name)

        if img_name and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image)
            except:
                image_tensor = torch.zeros(3, 224, 224)
        else:
            image_tensor = torch.zeros(3, 224, 224)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained("xlm-roberta-base")
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)

        self.fc1 = nn.Linear(768 + 256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        image_features = self.relu(self.image_model(image))
        fused = torch.cat((text_features, image_features), dim=1)
        x = self.dropout(self.relu(self.fc1(fused)))
        return self.out(x)