import torch
import torchvision
from torch import nn
from sklearn.metrics import roc_auc_score
import numpy as np


class basic_cnn(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (hidden_units, 1, 1)
            nn.Flatten(),                   # Output: (hidden_units,)
            nn.Linear(hidden_units, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_shape)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

"""
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = None):
    
    train_loss, train_auc = 0, 0
    
    model.to(device)
    
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_auc += roc_auc_score(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    train_loss /= len(dataloader)
    train_auc /= len(dataloader)
"""
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = None):
    
    train_loss = 0
    all_preds = []
    all_labels = []
    
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        train_loss += loss.item()
        
        if y_pred.shape[1] == 2:  # Binary classification
            # Use probability of positive class
            all_preds.extend(torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy())
        else:  # Multi-class
            all_preds.extend(torch.softmax(y_pred, dim=1).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    
    # Calculate AUC
    if len(set(all_labels)) > 1:
        if len(np.unique(all_labels)) == 2:  # Binary
            train_auc = roc_auc_score(all_labels, all_preds)
        else:  # Multi-class
            train_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    else:
        train_auc = 0.0
    
    return train_loss, train_auc

def val_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               device: torch.device = None):
    
    val_loss = 0
    all_preds = []
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            val_pred = model(X)
            loss = loss_fn(val_pred, y)
            
            val_loss += loss.item()
            
            if val_pred.shape[1] == 2:  # Binary classification
                # Use probability of positive class
                all_preds.extend(torch.softmax(val_pred, dim=1)[:, 1].detach().cpu().numpy())
            else:  # Multi-class
                all_preds.extend(torch.softmax(val_pred, dim=1).detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        val_loss /= len(dataloader)
        
        # Calculate AUC
        if len(set(all_labels)) > 1:
            if len(np.unique(all_labels)) == 2:  # Binary
                val_auc = roc_auc_score(all_labels, all_preds)
            else:  # Multi-class
                val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        else:
            val_auc = 0.0

    return val_loss, val_auc