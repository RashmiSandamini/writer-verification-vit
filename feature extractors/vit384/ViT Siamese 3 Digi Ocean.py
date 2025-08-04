#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
import pandas as pd
import os
from PIL import Image
import gdown
import zipfile
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = "Data_V9_ViT"

scaler = torch.amp.GradScaler('cuda')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[3]:


class LoadDataset(Dataset):
    def __init__(self, parquet_path, base_dir=BASE_DIR):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.base_dir = base_dir.rstrip("/")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def pt_path(rel_path: str) -> str:
            return os.path.join(self.base_dir, rel_path + ".pt")

        img1_path = pt_path(row["sample_1"])
        img2_path = pt_path(row["sample_2"])

        try:
            img1_tensor = torch.load(img1_path)
        except Exception as e:
            raise ValueError(f"Could not load tensor at {img1_path}: {e}")

        try:
            img2_tensor = torch.load(img2_path)
        except Exception as e:
            raise ValueError(f"Could not load tensor at {img2_path}: {e}")

        label = float(row["label"])
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return img1_tensor, img2_tensor, label_tensor


# In[ ]:


BATCH_SIZE = 128
VAL_BATCH_SIZE = 1024

train_dataset = LoadDataset(
    parquet_path=BASE_DIR + "/train.parquet"
)

val_dataset = LoadDataset(
    parquet_path=BASE_DIR + "/val.parquet"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
val_loader   = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# In[6]:


vit_name = "google/vit-base-patch16-384"
config = ViTConfig.from_pretrained(vit_name)
vit_backbone = ViTModel.from_pretrained(vit_name, config=config).to(device)

for param in vit_backbone.embeddings.parameters():
    param.requires_grad = False

for i in range(6):
    for param in vit_backbone.encoder.layer[i].parameters():
        param.requires_grad = False


# In[7]:


class ViTEmbedder(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        outputs = self.vit(pixel_values=x)   
        return outputs.pooler_output    

vit_embedder = ViTEmbedder(vit_backbone).to(device)


# In[ ]:


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=[1024, 512, 256], dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
            ]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)          
        return F.normalize(z, p=2, dim=1)

proj_head = ProjectionHead(
    in_dim=768,
    hidden_dims=[512, 256, 128],
    dropout=0.1
).to(device)


# In[9]:


class EuclideanDistance(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff_sq = (x - y).pow(2)
        sum_sq = diff_sq.sum(dim=1, keepdim=True)
        sum_sq = torch.clamp(sum_sq, min=self.eps)
        return torch.sqrt(sum_sq)


# In[10]:


class SiameseViT(nn.Module):
    def __init__(self, embedder, head):
        super().__init__()
        self.embedder = embedder
        self.head = head
        self.distance = EuclideanDistance()

    def forward_once(self, img):
        cls_emb = self.embedder(img)
        emb256 = self.head(cls_emb)
        return emb256

    def forward(self, img_a, img_b):
        emb_a = self.forward_once(img_a)
        emb_b = self.forward_once(img_b)
        dist = self.distance(emb_a, emb_b)
        return dist

siamese_model = SiameseViT(vit_embedder, proj_head).to(device)


# In[11]:


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):

        label = label.view(-1, 1) 
        pos_loss = (1.0 - label) * torch.pow(dist, 2)
        neg_loss = label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(pos_loss + neg_loss)
        return loss

criterion = ContrastiveLoss(margin=1.0).to(device)


# In[12]:


optimizer = torch.optim.Adam(
    siamese_model.parameters(),
    lr=1e-5,
)


# In[13]:


MODEL_DIR = 'siamese_checkpoints'
HIST_DIR  = 'histograms'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

best_val_auc = 0.0

def plot_and_save_histogram(dists, labels, epoch):
    neg = dists[labels == 0]
    pos = dists[labels == 1]

    plt.figure(figsize=(6,4))
    plt.hist(neg, bins=50, alpha=0.6, label="Same Writer")
    plt.hist(pos, bins=50, alpha=0.6, label="Different Writer")
    plt.title(f"Distance dist. @ epoch {epoch}")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Count")
    plt.legend()
    fname = os.path.join(HIST_DIR, f"epoch_{epoch:02d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    global best_val_auc

    for epoch in range(1, num_epochs+1):

        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc, val_dists, val_labels = validate(model, val_loader, criterion)

        epoch_path = os.path.join(MODEL_DIR, f'epoch_{epoch:02d}.pt')
        torch.save(model.state_dict(), epoch_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_path = os.path.join(MODEL_DIR, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            print(f"→ New best AUC {val_auc:.4f}, saved to {best_path}")
        print(f"Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}")
        hist_file = plot_and_save_histogram(val_dists, val_labels, epoch)
        # img = plt.imread(hist_file)
        # plt.imshow(img)
        # plt.title(f"Distnace Distribution: {epoch}")
        # plt.axis("off")
        # plt.show()
        # writer.add_image('DistanceHist', img, epoch, dataformats='HWC')



# In[14]:


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_samples = 0

    train_auc_metric = AUROC(task="binary")

    pbar = tqdm(dataloader, desc="Train batches", leave=False)

    for img_a, img_b, labels in pbar:
        img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
        labels = labels.squeeze(1).long()   # shape [B], dtype=int64

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            dist = model(img_a, img_b)
            loss = criterion(dist, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = img_a.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        train_auc_metric.update(dist.view(-1), labels)
        epoch_loss_so_far = running_loss / total_samples
        epoch_auc_so_far = train_auc_metric.compute().item()

        pbar.set_postfix({
            "loss": f"{epoch_loss_so_far:.4f}",
            "auc":  f"{epoch_auc_so_far:.4f}"
        })

    train_loss = running_loss / len(dataloader.dataset)
    train_auc  = train_auc_metric.compute().item()
    return train_loss, train_auc


# ─── 2. VALIDATION ─────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = total = 0
    all_dists, all_labels = [], []

    for img_a, img_b, labels in tqdm(dataloader, desc="Val batches", leave=False):
        img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
        labels = labels.squeeze(1).long()

        dist = model(img_a, img_b)[:, 0]
        loss = criterion(dist, labels)

        batch = img_a.size(0)
        running_loss += loss.item() * batch
        total        += batch

        all_dists.append(dist.cpu())
        all_labels.append(labels.cpu())

    val_loss = running_loss / total
    dists    = torch.cat(all_dists).numpy()
    labs     = torch.cat(all_labels).numpy()
    val_auc  = roc_auc_score(labs, dists)

    return val_loss, val_auc, dists, labs


# In[15]:


num_epochs = 10
train(
    model      = siamese_model,
    train_loader = train_loader,
    val_loader   = val_loader,
    optimizer    = optimizer,
    criterion    = criterion,
    num_epochs   = num_epochs,
    device       = device
)

