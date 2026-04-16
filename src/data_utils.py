"""
Data loading and transformation utilities.
"""
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np


class FeatureExtractor:
    """Extracts deep features from images using a pre-trained backbone."""
    
    def __init__(self, device):
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()

    def extract_features(self, loader, device):
        """Extract features from all images in loader."""
        features = []
        labels = []
        with torch.no_grad():
            for images, lbls in tqdm(loader, desc="Extracting Features"):
                images = images.to(device)
                outputs = self.model(images)
                features.append(outputs.cpu().numpy())
                labels.extend(lbls.numpy())
        return np.concatenate(features), np.array(labels)


def gpu_check():
    """Returns CUDA device if available, else exits."""
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("No GPU found, exiting program.")
        exit(1)


def get_vanilla_transform():
    """Basic transform for feature extraction."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_advanced_transform():
    """Advanced data augmentation transform."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3, fill=0),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_image_transform(mode='advanced'):
    """Get image transform by mode: 'vanilla' or 'advanced'."""
    if mode == 'vanilla':
        return get_vanilla_transform()
    return get_advanced_transform()


def load_dataset(DATASET_PATH, transform, batch_size=32):
    """Load dataset and create data loaders."""
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'num_classes': num_classes,
        'classes': dataset.classes,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }