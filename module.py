import torch
import torchvision
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path


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

def find_optimal_threshold(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='weighted')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1

def test_with_threshold(model: nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        loss_fn: nn.Module,
                        device: torch.device = None,
                        threshold: float = 0.5,
                        verbose: bool = True):
    """
    Test the model with a custom probability threshold for binary classification.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for test data
        loss_fn: Loss function
        device: Device to run on (cuda/cpu)
        threshold: Probability threshold for positive class (default: 0.5)
        verbose: Whether to print detailed results
    
    Returns:
        test_loss: Average loss on test set
        test_auc: AUC score
        accuracy: Accuracy score
        all_preds: All predictions
        all_labels: All ground truth labels
    """
    
    test_loss = 0
    all_preds_probs = []  # Store probabilities
    all_preds = []        # Store final predictions after threshold
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # Get probabilities
            if test_pred.shape[1] == 2:  # Binary classification
                probs = torch.softmax(test_pred, dim=1)
                # Get probability of positive class (class 1)
                prob_positive = probs[:, 1].detach().cpu().numpy()
                all_preds_probs.extend(prob_positive)
                
                # Apply custom threshold
                batch_preds = (prob_positive >= threshold).astype(int)
                all_preds.extend(batch_preds)
            else:  # Multi-class (fallback to argmax)
                probs = torch.softmax(test_pred, dim=1)
                all_preds_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            
            all_labels.extend(y.cpu().numpy())
    
    # Calculate average loss
    test_loss /= len(dataloader)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_preds_probs = np.array(all_preds_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC (using probabilities, not thresholded predictions)
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
        print(f"TEST RESULTS (Threshold = {threshold:.2f})")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("="*50)
    
    return test_loss, test_auc, accuracy, all_preds, all_labels


def print_result_simple(result):
    """
    Nicely print a prediction result.
    """
    print("\n" + "="*60)
    print(f"Image: {result['image']}")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability_percent']} (threshold: 0.55)")
    print(f"Confidence: {result['confidence']}")
    print(f"Status: {result['review_message']}")
    print("="*60)


def save_results_simple(results, output_file='predictions.txt'):
    """
    Save results to a simple text file.
    """
    with open(output_file, 'w') as f:
        f.write("PNEUMONIA DETECTION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        pneumonia_cases = [r for r in results if r['prediction'] == "PNEUMONIA"]
        normal_cases = [r for r in results if r['prediction'] == "NORMAL"]
        review_cases = [r for r in results if r['needs_review']]
        
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Pneumonia detected: {len(pneumonia_cases)}\n")
        f.write(f"Normal: {len(normal_cases)}\n")
        f.write(f"Needs review: {len(review_cases)}\n\n")
        
        if pneumonia_cases:
            f.write("PNEUMONIA CASES:\n")
            f.write("-"*40 + "\n")
            for r in pneumonia_cases:
                f.write(f"{r['image']}: {r['probability_percent']} - {r['confidence']}\n")
            f.write("\n")
        
        if review_cases:
            f.write("CASES NEEDING REVIEW:\n")
            f.write("-"*40 + "\n")
            for r in review_cases:
                f.write(f"{r['image']}: {r['prediction']} ({r['probability_percent']})\n")
    
    print(f"\n✅ Results saved to {output_file}")


def prepare_image_for_model(image_path, model_type='custom'):
    """
    Prepare image for model input.
    
    Args:
        image_path: Path to image or PIL Image
        model_type: 'custom' for grayscale (1-channel) or 'resnet152' for 3-channel RGB
    """
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')
    
    if model_type == 'custom':
        # For your original custom grayscale model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif model_type == 'resnet152':
        # For your trained ResNet152 model
        # Note: This matches your training transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Generic pretrained model (like default ResNet)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transform(img).unsqueeze(0)
    return image_tensor


def get_pneumonia_probability(model, image_tensor, device, model_type='custom'):

    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.inference_mode():
        output = model(image_tensor)
        
        # Both your custom model and ResNet152 have 2 output classes
        probabilities = torch.softmax(output, dim=1)
        pneumonia_prob = probabilities[:, 1].cpu().numpy()[0]
    
    return float(pneumonia_prob)


def simple_predict(model, image_path, device, threshold=0.55, model_type='custom'):

    image_tensor = prepare_image_for_model(image_path, model_type)
    prob = get_pneumonia_probability(model, image_tensor, device, model_type)
    
    # Rest of your function remains the same...
    prediction = "PNEUMONIA" if prob >= threshold else "NORMAL"
    
    if prediction == "PNEUMONIA":
        if prob >= 0.90:
            confidence = "VERY HIGH - Definitely pneumonia"
        elif prob >= 0.75:
            confidence = "HIGH - Likely pneumonia"
        elif prob >= threshold:
            confidence = "MEDIUM - Possible pneumonia, please review"
        else:
            confidence = "LOW - Just above threshold"
    else:  # NORMAL
        if prob <= 0.10:
            confidence = "VERY HIGH - Definitely normal"
        elif prob <= 0.25:
            confidence = "HIGH - Likely normal"
        elif prob < threshold:
            confidence = "MEDIUM - Possibly normal, please review"
        else:
            confidence = "LOW - Just below threshold"
    
    needs_review = 0.25 < prob < 0.75
    
    result = {
        'image': str(image_path),
        'prediction': prediction,
        'probability': round(prob, 3),
        'probability_percent': f"{prob:.1%}",
        'confidence': confidence,
        'needs_review': needs_review,
        'review_message': "⚠️ Please review this case" if needs_review else "✓ OK"
    }
    
    return result


def example_usage():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    while True :
    
        print(f"Using device: {device}")
        print("\nSelect model type:")
        print("1. Custom CNN (grayscale, 1-channel)")
        print("2. ResNet152 (3-channel, from your training)")
        print("0. Exit program")
        
        choice = input("Enter choice (1, 2 or 0): ").strip()
        
        if choice == '0':
            print("\n👋 Exiting program. Goodbye!")
            return
        
        elif choice == '2':
            import torchvision.models as models
            
            model_res152 = models.resnet152(weights=None)
            num_features = model_res152.fc.in_features
            model_res152.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True), 
                nn.Dropout(0.4),
                nn.Linear(256, 2)
            )
            
            model_path = 'models/model_resnet152.pth'
            if os.path.exists(model_path):
                model_res152.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ ResNet152 model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                return
            
            model_to_use = model_res152
            model_type = 'resnet152'
            OPTIMAL_THRESHOLD = 0.5
            
        elif choice == '1':
            # Load your custom CNN model
            from module import improved_cnn
            
            custom_model = improved_cnn(input_shape=1, hidden_units=10, output_shape=2)
            model_path = 'models/model1_improved_cnn.pth'
            
            if os.path.exists(model_path):
                custom_model.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ Custom CNN model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                return
            
            model_to_use = custom_model
            model_type = 'custom'
            OPTIMAL_THRESHOLD = 0.55
        
        model_to_use.to(device)
        model_to_use.eval()
        
        # Single image prediction
        single_image_pred = int(input("\nEnter 1 for single image prediction, 0 for batch prediction: "))
        
        if single_image_pred == 1:
            print("\n" + "="*60)
            print("SINGLE IMAGE PREDICTION")
            print("="*60)
            
            while True:
                print("\n" + "-"*40)
                user_input = input("Enter image path (or '0' to exit): ")
                
                if user_input == '0':
                    print("Exiting single image prediction...")
                    break
                
                if os.path.exists(user_input):
                    result = simple_predict(model_to_use, user_input, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    print_result_simple(result)
                else:
                    print(f"❌ File not found: {user_input}")
                    print("Please check the path and try again.")
        else:
            # Batch prediction
            print("\n" + "="*60)
            print("BATCH PREDICTION")
            print("="*60)
            
            test_normal_dir = "data/reorganized_chest_xray/test/NORMAL"
            test_pneumonia_dir = "data/reorganized_chest_xray/test/PNEUMONIA"
            
            all_results = []
            
            if os.path.exists(test_normal_dir):
                normal_images = list(Path(test_normal_dir).glob("*.jpeg"))[:5]  # First 5 normal images
                for img_path in normal_images:
                    result = simple_predict(model_to_use, img_path, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    all_results.append(result)
            
            if os.path.exists(test_pneumonia_dir):
                pneumonia_images = list(Path(test_pneumonia_dir).glob("*.jpeg"))[:5]  # First 5 pneumonia images
                for img_path in pneumonia_images:
                    result = simple_predict(model_to_use, img_path, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    all_results.append(result)
            
            # Print summary
            print(f"\nProcessed {len(all_results)} images:")
            for r in all_results:
                print(f"  {Path(r['image']).name:<30} → {r['prediction']:<10} ({r['probability_percent']}) {r['review_message']}")
            
            # Save results
            save_results_simple(all_results, f'prediction_samples/predictions_{model_type}.txt')

if __name__ == "__main__":
    example_usage()
