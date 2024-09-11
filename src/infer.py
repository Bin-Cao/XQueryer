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

def get_acc(cls, label):
    correct_cnt = (cls.argmax(1) == label.int()).sum().item()
    cls_acc = correct_cnt / cls.shape[0]
    return cls_acc, correct_cnt

def run_one_epoch(model, dataloader, device):
    model.eval()

    epoch_loss, cls_acc = 0., 0.
    correct_cnt, total_cnt = 0, 0

    # Initialize lists for storing true and predicted labels
    all_preds = []
    all_labels = []

    pbar = tqdm(total=len(dataloader.dataset), desc='Evaluating... ', unit='data')
    iters = len(dataloader)
    criterion = torch.nn.CrossEntropyLoss()  

    for batch in dataloader:
        intensity = batch['intensity'].to(device)
        label_cls = batch['id'].to(device)
        element = batch['element'].to(device)

        with torch.no_grad():
            with autocast():
                logits = model(intensity, element)

        pbar.update(len(intensity))

        _cls_acc, correct = get_acc(logits, label_cls)
        cls_acc += _cls_acc

        correct_cnt += correct
        total_cnt += len(intensity)

        # Calculate loss
        loss = criterion(logits, label_cls)
        epoch_loss += loss.item()

        # Store predictions and true labels
        preds = logits.argmax(1).cpu().numpy()
        labels = label_cls.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    pbar.close()

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate precision, recall, F1 score
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss / iters, cls_acc * 100 / iters, correct_cnt, total_cnt, precision, recall, f1

def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = Xmodel(embed_dim=3500, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.load_path, map_location=device)['model'])
    model.to(device)
    model.eval()
    print('Loaded model from {}'.format(args.load_path))

    valset = ASEDataset(args.data_dir, args.atom_embed)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    loss_val, acc_val, correct_cnt, total_cnt, precision, recall, f1 = run_one_epoch(model, val_loader, device)

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
