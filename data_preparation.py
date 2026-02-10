import os
from pathlib import Path

def quick_count(data_dir):
    root = Path(data_dir)
    counts = {}
    
    for split in ['train', 'val', 'test']:
        split_path = root / split
        counts[split] = {}
        
        for cls in ['NORMAL', 'PNEUMONIA']:
            cls_path = split_path / cls
            if cls_path.exists():
                image_files = [f for f in cls_path.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                counts[split][cls] = len(image_files)
            else:
                counts[split][cls] = 0
                
    return counts

data_dir = "./data/chest_xray"
counts = quick_count(data_dir)

print("Dataset Distribution:")
print("-" * 40)

for split in counts:
    normal = counts[split].get('NORMAL', 0)
    pneumonia = counts[split].get('PNEUMONIA', 0)
    total = normal + pneumonia
    
    if total > 0:
        normal_pct = (normal / total) * 100
        pneumonia_pct = (pneumonia / total) * 100
        
        print(f"{split.upper():<10} NORMAL: {normal:>4} ({normal_pct:5.1f}%)  |  PNEUMONIA: {pneumonia:>4} ({pneumonia_pct:5.1f}%)")
    else:
        print(f"{split.upper():<10} No images found")
