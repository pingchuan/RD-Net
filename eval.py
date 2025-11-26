import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from models.ResUNet import ResNet34U_f
from utils.metrics import Metrics,evaluate
from torchvision import transforms
from PIL import Image
from utils.transform import *

def get_transform():
    return transforms.Compose([
        Resize((320, 320)),
        ToTensor(),
    ])


class ImageFolderDataset(Dataset):
    """Dataset for loading image and mask pairs from folders."""
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.image_files[index])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size

        mask_name = self.image_files[index]
        mask_path = os.path.join(self.mask_folder, mask_name)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            data = {'image': img, 'label': mask}
            data = self.transform(data)
            img_resized, mask_resized = data['image'], data['label']
        else:
            img_resized = img
            mask_resized = mask

        return {
            'image': img_resized,
            'mask': mask_resized,
            'filename': self.image_files[index],
            'original_size': original_size
        }


def validate(model, valid_dataloader, device):
    """Validate the model and compute metrics."""
    model.eval()
    metrics = Metrics(['ACC_overall', 'Dice', 'IoU'])

    with torch.no_grad():
        for data in tqdm(valid_dataloader, desc="Validating"):
            img = data['image'].to(device)
            gt = data['mask'].to(device)

            output = model(img)
            metrics.update(**evaluate(output, gt))

    metrics_result = metrics.mean()
    print(
        f"{'Valid':^10s}  : {', '.join([f'{k}:{v * 100:.2f}%' for k, v in metrics_result.items()])}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for polyp segmentation")
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the test image folder')
    parser.add_argument('--mask_folder', type=str, required=True, help='Path to the ground truth mask folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = ResNet34U_f(num_classes=1)
    if os.path.exists(args.weights):
        print(f"Loading pretrained weights from {args.weights}")
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    model = model.to(device)

    # Prepare dataset and dataloader
    transform = get_transform()
    valid_dataset = ImageFolderDataset(
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        transform=transform
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Run validation
    validate(model, valid_dataloader, device)
