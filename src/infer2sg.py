import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.dataset import ASEDataset
from model.XQueryer import Xmodel
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score
import json

# Load the dictionary from the JSON file
with open('entries_dict.json', 'r') as file:
    entries_dict = json.load(file)

def run_one_epoch(model, dataloader, device, entries_dict):
    model.eval()  # Set the model to evaluation mode

    epoch_loss = 0.0
    correct_cnt = 0
    total_cnt = 0

    # Initialize lists for storing true and predicted sg values
    all_preds = []
    all_labels = []

    # Progress bar for tracking evaluation
    pbar = tqdm(total=len(dataloader.dataset), desc='Evaluating... ', unit='data')
    iters = len(dataloader)
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        # Move input data to the specified device
        intensity = batch['intensity'].to(device)
        label_cls = batch['id'].to(device)
        element = batch['element'].to(device)

        with torch.no_grad():
            with autocast():
                logits = model(intensity, element)

        # Update progress bar
        pbar.update(len(intensity))

        # Calculate loss
        loss = criterion(logits, label_cls)
        epoch_loss += loss.item()

        # Get predictions
        preds = logits.argmax(1)

        # Map `preds` and `label_cls` to 'sg' values based on 'key'
        pred_sg = [entries_dict[str(key.item())]['sg'] for key in preds]
        label_sg = [entries_dict[str(key.item())]['sg'] for key in label_cls]

        # Store mapped `sg` values for metric calculations
        all_preds.extend(pred_sg)
        all_labels.extend(label_sg)

        # Count correct predictions
        correct_sg_cnt = sum(p == l for p, l in zip(pred_sg, label_sg))
        correct_cnt += correct_sg_cnt
        total_cnt += label_cls.size(0)

    pbar.close()

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate overall accuracy
    accuracy = correct_cnt / total_cnt if total_cnt > 0 else 0

    # Calculate precision, recall, and F1 score for `sg` values
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Return metrics for the epoch
    return epoch_loss / iters, accuracy, correct_cnt, total_cnt, precision, recall, f1



def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = Xmodel(embed_dim=3500, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.load_path, map_location=device)['model'])
    model.to(device)
    model.eval()
    print('Loaded model from {}'.format(args.load_path))

    valset = ASEDataset(args.data_dir, args.atom_embed)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    loss_val, acc_val, correct_cnt, total_cnt, precision, recall, f1 = run_one_epoch(model, val_loader, device,entries_dict)

    print("Validation Loss: ", loss_val)
    print("Validation Accuracy: ", acc_val)
    print(f"Accuracy: {round(correct_cnt / total_cnt * 100, 3)}%  ({correct_cnt}/{total_cnt})")
    print(f"Precision: {round(precision * 100, 3)}%")
    print(f"Recall: {round(recall * 100, 3)}%")
    print(f"F1 Score: {round(f1 * 100, 3)}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, choices=['cuda:0', 'cpu'])
    parser.add_argument('--data_dir', nargs='+', default=['/data/cb_dataset/test.db'], type=str,
                        help='List of test data directories (space-separated). Example: --data_dir_train /path/to/test1.db /path/to/test2.db')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--atom_embed', default=True, type=bool)
    parser.add_argument('--load_path', default='/home/cb/XRDS/XQueryer/output/2024-09-09_1117/checkpoints/checkpoint_0010.pth', type=str,
                        help='Path to load pretrained single-phase identification model')
    parser.add_argument('--num_classes', default=100315, type=int)

    args = parser.parse_args()
    main()
    print('THE END')
