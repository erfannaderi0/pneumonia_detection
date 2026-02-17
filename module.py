import torch
import torchvision
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import numpy as np


class improved_cnn(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        dropout_rate = 0.2
        
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(hidden_units*4, hidden_units*8, 3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, 3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units*8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_shape)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.dropout2(x)
        x = self.block3(x)
        x = self.dropout3(x)
        x = self.block4(x)
        x = self.dropout4(x)
        x = self.classifier(x)
        return x


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


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device = None,
              verbose: bool = True):
    
    test_loss = 0
    all_preds = []
    all_preds_probs = []
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            if test_pred.shape[1] == 2:  # Binary classification
                probs = torch.softmax(test_pred, dim=1)

                all_preds_probs.extend(probs[:, 1].detach().cpu().numpy())

                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            else:  # Multi-class
                probs = torch.softmax(test_pred, dim=1)
                all_preds_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            
            all_labels.extend(y.cpu().numpy())
    
    test_loss /= len(dataloader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    if len(np.unique(all_labels)) > 1:
        if len(np.unique(all_labels)) == 2:  # Binary
            test_auc = roc_auc_score(all_labels, all_preds_probs)
        else:  # Multi-class
            test_auc = roc_auc_score(all_labels, all_preds_probs, multi_class='ovr')
    else:
        test_auc = 0.0
    
    # Print detailed metrics if verbose
    if verbose:
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("="*50)
    
    return test_loss, test_auc, accuracy, all_preds, all_labels
