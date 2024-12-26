import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio.transforms as T
from tqdm import tqdm
from torch.optim.lr_scheduler import CyclicLR

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    feature_extractor = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=80
        ),
        T.AmplitudeToDB(),
        nn.Lambda(lambda x: x - torch.mean(x, dim=-1, keepdim=True))
    ).to(device)
    
    best_acc = 0
    patience = 10  # Early stopping 용
    patience_counter = 0
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)

    for epoch in epoch_pbar:
        # Training
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
        for batch_idx, (audio, labels) in enumerate(train_pbar):
            features = feature_extractor(audio.to(device))
            embeddings = model(features)
            
            loss = criterion(embeddings, labels.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({"loss":f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = eval_model(model, val_loader, device, criterion, feature_extractor)
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'Curr_lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        if scheduler is not None:
            scheduler.step(val_acc)

        print(f"Epoch {epoch}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Current LR : {optimizer.param_groups[0]['lr']:.6f}")
        
        # 모델 저장 (검증 정확도 기준)
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, 'best_ecapa_tdnn.pth')

        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

def eval_model(model, val_loader, device, criterion, feature_extractor):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        eval_pbar = tqdm(val_loader, desc='Evaluating', position=1, leave=False)
        for batch_idx, (audio, labels) in enumerate(eval_pbar):
            features = feature_extractor(audio.to(device))
            embeddings = model(features)
            
            # AAMsoftmax loss 계산
            loss = criterion(embeddings, labels.to(device))
            total_loss += loss.item()
            
            # 정확도 계산을 위한 예측
            # cosine similarity 기반으로 가장 가까운 화자 찾기
            similarity = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
            predictions = torch.argmax(similarity, dim=1)
            
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)

            eval_pbar.set_postfix({'Val loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, filename)