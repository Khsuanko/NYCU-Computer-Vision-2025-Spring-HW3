import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

from dataset import CellSegmentationDataset
from model import get_instance_segmentation_model
import configs

def collate_fn(batch):
    return tuple(zip(*batch))

def save_loss_curve(train_losses, save_path):
    """
    Save training loss curve to file.
    
    Args:
        train_losses (list): List of training losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def main():

    train_loss_history = []

    dataset = CellSegmentationDataset(configs.train_dir, transforms=T.ToTensor())
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=configs.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_instance_segmentation_model(configs.num_classes)
    model.to(configs.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=configs.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(configs.num_epochs):
        model.train()
        total_loss = 0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.num_epochs}")
        for images, targets in pbar:
            images = [img.to(configs.device) for img in images]
            targets = [{k: v.to(configs.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        #print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(train_loader):.4f}")
        lr_scheduler.step()

    torch.save(model.state_dict(), configs.save_path)
    print(f"Model saved to {configs.save_path}")

    # Create output directory if needed
    os.makedirs('plots', exist_ok=True)
    
    # Save final loss curve
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/training_loss_{timestamp}.png"
    save_loss_curve(train_loss_history, plot_path)

if __name__ == "__main__":
    main()
