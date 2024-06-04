import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.dataset import ASEDataset
from model.XQueryer import Xmodel




def get_acc(cls, label):
    correct_cnt = sum(cls.argmax(1) == label.int())
    cls_acc = correct_cnt / cls.shape[0]
    return cls_acc, correct_cnt


def run_one_epoch(model, dataloader):
    model.eval()

    epoch_loss, cls_acc = 0, 0
    correct_cnt, total_cnt = 0, 0
    pbar = tqdm(total=len(dataloader.dataset), desc='Evaluating... ', unit='data')
    iters = len(dataloader)
    for i, batch in enumerate(dataloader):
        # readin data
        # latt_dis = batch['latt_dis'].to(args.device)
        intensity = batch['intensity'].to(args.device)
        label_cls = batch['id'].to(args.device)
        element = batch['element'].to(args.device)

        with torch.no_grad():
            logits = model(intensity,element)
            logits.to(args.device)

        pbar.update(len(intensity))

        _cls_acc, correct = get_acc(logits, label_cls)
        cls_acc += _cls_acc.item()

        correct_cnt += correct.item()
        total_cnt += len(intensity)

        preds = logits.argmax(1)

    return epoch_loss / iters, cls_acc * 100 / iters, correct_cnt, total_cnt, preds


def main():
    model = Xmodel(embed_dim=3500, num_classes=args.num_classes)

    loaded = torch.load(args.load_path)
    model.load_state_dict(loaded['model'])
    model.to(args.device)
    model.eval()
    print('loaded model from {}'.format(args.load_path))

    print(model)

    valset = ASEDataset(args.data_dir,args.atom_embed)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    loss_val, acc_val, correct_cnt, total_cnt, preds = run_one_epoch(model, val_loader)

    print("loss_val: ", loss_val)
    print("acc_val: ", acc_val)
    print("{}%  ({}/{})".format(round(correct_cnt / total_cnt, 5) * 100, correct_cnt, total_cnt))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--data_dir', default=['/data/cb_dataset/test.db'], type=str)
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        )
    parser.add_argument('--num_workers', default=16, type=int, metavar='N',
                       )
    parser.add_argument('--atom_embed', default=True, type=bool)
    parser.add_argument('--load_path', default='pretrained/checkpoint_0010.pth', type=str,
                        help='path to load pretrained single-phase identification model')
    parser.add_argument('--num_classes', default=100315, type=int, metavar='N')

    args = parser.parse_args()

    main()

    print('THE END')