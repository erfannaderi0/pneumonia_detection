import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import module
from module import *
from tqdm.auto import tqdm


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(42)

    # ImageNet statistics
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])
    }

    data_dir = './data/reorganized_chest_xray'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    train_dataloader = torch.utils.data.DataLoader(image_datasets['train'],
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=4)
    val_test_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=32,
                                                           shuffle=False,
                                                           num_workers=4)
                            for x in ['val', 'test']}
    val_dataloader = val_test_dataloaders['val']
    test_dataloader = val_test_dataloaders['test']
    if not os.path.exists('models/model_resnet152.pth'):
        model_res152 = models.resnet152(weights='IMAGENET1K_V1')

        for params in model_res152.parameters():
            params.requires_grad = False

        num_features = model_res152.fc.in_features

        model_res152.fc = nn.Sequential(nn.Linear(num_features, 256),
                                        nn.ReLU(inplace=True), 
                                        nn.Dropout(0.4),
                                        nn.Linear(256, 2))

        #=====================================================================
        #=                        TRAINING PHASE                             =
        #=====================================================================
        model_res152 = model_res152.to(device)
        
        optimizer_ft = optim.Adam(model_res152.fc.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=3)

        loss_fn = nn.CrossEntropyLoss()

        epochs = 10

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in tqdm(range(epochs)):
            train_loss, train_auc = train_step(model=model_res152,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer_ft,
                                            device=device)
                
            val_loss, val_auc = val_step(model=model_res152,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
                
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                
            exp_lr_scheduler.step(val_loss)
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model_res152.state_dict(), 'models/model_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        for param in model_res152.parameters():
            param.requires_grad = True

        optimizer_ft = optim.Adam(model_res152.parameters(), lr=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        for epoch in tqdm(range(epochs)):
            
            train_loss, train_auc = train_step(model=model_res152,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer_ft,
                                            device=device)
                
            val_loss, val_auc = val_step(model=model_res152,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
                
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                
            exp_lr_scheduler.step(val_loss)
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model_res152.state_dict(), 'models/model_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    else:
        loss_fn = nn.CrossEntropyLoss()
        
        model_res152 = models.resnet152(weights=None)
        
        num_features = model_res152.fc.in_features
        model_res152.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.4),
            nn.Linear(256, 2))
        
        model_path = 'models/model_resnet152.pth'
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model_res152.load_state_dict(state_dict)
            print("✅ ResNet152 model loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
        
        model_res152.to(device)
        
        model_resnet_result = test_step(model=model_res152,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

    example_usage()
