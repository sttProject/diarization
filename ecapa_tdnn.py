import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import DataLoader

from dataset import AMIDataset
from model import ECAPA_TDNN
from loss import AAMsoftmax
from utils import train_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# device 설정 부분 수정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


train_dataset = AMIDataset(
    split='train',
    max_duration=3.0,
    overlap=1.5,
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=AMIDataset.collate_fn
)

val_dataset = AMIDataset(
    split='validation',
    max_duration=3.0,
    overlap=1.5,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=AMIDataset.collate_fn
)

model = ECAPA_TDNN().to(device)
criterion = AAMsoftmax(n_class=155, m=0.2, s=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Cyclical Learning Rate 설정
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-8,  # 최소 학습률
    max_lr=1e-3,   # 최대 학습률
    step_size_up=2000,  # 반주기 스텝 수
    mode='triangular'  # 삼각형 정책 사용
)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device=device)
