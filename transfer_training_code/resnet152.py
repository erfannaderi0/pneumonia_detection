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
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(42)

    # ImageNet statistics
    xray_mean = [0.482, 0.482, 0.482]
    xray_std  = [0.234, 0.234, 0.234]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)])
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
    if not os.path.exists('models/model2_resnet152.pth'):
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

        y_train = [label for _, label in image_datasets['train'].samples]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

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
                torch.save(model_res152.state_dict(), 'models/model2_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        for param in model_res152.parameters():
            param.requires_grad = False  # re-freeze everything first
        for param in model_res152.layer4.parameters():
            param.requires_grad = True
        for param in model_res152.layer3.parameters():
            param.requires_grad = True
        for param in model_res152.fc.parameters():
            param.requires_grad = True
        print("\nPhase 2: Unfroze layer3, layer4, fc — fine-tuning with lr=1e-5")

        optimizer_ft = optim.Adam(model_res152.parameters(), lr=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        
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
                
            exp_lr_scheduler.step()
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model_res152.state_dict(), 'models/model2_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    else:
        y_train = [label for _, label in image_datasets['train'].samples]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        model_res152 = models.resnet152(weights=None)
        
        num_features = model_res152.fc.in_features
        model_res152.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.4),
            nn.Linear(256, 2))
        
        model_path = 'models/model2_resnet152.pth'
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model_res152.load_state_dict(state_dict)
            print("✅ ResNet152 model loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
        
        model_res152.to(device)
        '''
        model_resnet_result = test_step(model=model_res152,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        '''

        # Initialize plotter
        plotter = Plot(style='seaborn-v0_8-darkgrid', figsize=(12, 8))

        # After running test_step
        test_loss, test_auc, accuracy, all_preds, all_labels, all_preds_probs = test_step(model_res152,
                                                                                          test_dataloader,
                                                                                          loss_fn,
                                                                                          device)

        # Get prediction probabilities (you'd need to modify test_step to return them)
        # all_preds_probs should be returned from test_step

        # 1. Plot confusion matrix
        plotter.plot_confusion_matrix(all_labels,
                                      all_preds,
                                      class_names=['Class 0', 'Class 1'],
                                      normalize=False,
                                      title='Test Set Confusion Matrix',
                                      save_path='pictures/plot_confusion_matrix_res152')

        # 2. Plot ROC curves
        plotter.plot_roc_curves(all_labels,
                                all_preds_probs,
                                n_classes=2,
                                class_names=['Class 0', 'Class 1'],
                                title='ROC Curves - Test Set',
                                save_path='pictures/plot_roc_curves')

        # 3. Plot Precision-Recall curves
        plotter.plot_precision_recall_curves(all_labels,
                                             all_preds_probs,
                                             n_classes=2,
                                             class_names=['Class 0', 'Class 1'],
                                             save_path='pictures/plot_oercision_recall_curves_res152')

        # 4. Learning curves (if you stored losses during training)
        # train_losses = [...]  # List of training losses per epoch
        # val_losses = [...]    # List of validation losses per epoch
        # plotter.plot_learning_curves(train_losses, val_losses)

        # 5. Error analysis
        plotter.plot_error_analysis(all_labels,
                                    all_preds,
                                    feature_names=['feature1', 'feature2'],
                                    save_path='pictures/plot_error_analysis_res152')
    
    example_usage()
