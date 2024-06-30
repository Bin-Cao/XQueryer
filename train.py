import argparse
import math
import os
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ASEDataset
from model.XQueryer import Xmodel
from model.focal_loss import FocalLoss
from util.logger import Logger

def get_acc(cls: torch.Tensor, label: torch.Tensor) -> float:
    cls_acc = (cls.argmax(1) == label.int()).float().mean().item()
    return cls_acc

def run_one_epoch(model, dataloader, criterion, optimizer, epoch, mode):
    if mode == 'Train':
        model.train()
        criterion.train()
        desc = 'Training... '
    else:
        model.eval()
        criterion.eval()
        desc = 'Evaluating... '

    epoch_loss, cls_acc = 0, 0
    if args.progress_bar:
        pbar = tqdm(total=len(dataloader.dataset), desc=desc, unit='data')
    iters = len(dataloader)

    for i, batch in enumerate(dataloader):
        # Read data
        intensity = batch['intensity'].to(device)
        label_cls = batch['id'].to(device)
        element = batch['element'].to(device)

        if mode == 'Train':
            adjust_learning_rate_withWarmup(optimizer, epoch + i / iters, args)
            with torch.cuda.amp.autocast():
                logits = model(intensity, element)
                loss = criterion(logits, label_cls.long())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(intensity, element)
                loss = criterion(logits, label_cls.long())

        epoch_loss += loss.item()
        if args.progress_bar:
            pbar.update(len(intensity))
            pbar.set_postfix(loss=loss.item())

        _cls_acc = get_acc(logits, label_cls)
        cls_acc += _cls_acc

    if args.progress_bar:
        pbar.close()

    return epoch_loss / iters, cls_acc * 100 / iters

def print_log(epoch: int, loss_train: float, loss_val: float, acc_train: float, acc_val: float, lr: float):
    if rank == 0:
        log.printlog(f'---------------- Epoch {epoch} ----------------')
        log.printlog(f'loss_train : {round(loss_train, 4)}')
        log.printlog(f'loss_val   : {round(loss_val, 4)}')
        log.printlog(f'acc_train  : {round(acc_train, 4)}%')
        log.printlog(f'acc_val    : {round(acc_val, 4)}%')

        log.train_writer.add_scalar('loss', loss_train, epoch)
        log.val_writer.add_scalar('loss', loss_val, epoch)
        log.train_writer.add_scalar('acc', acc_train, epoch)
        log.val_writer.add_scalar('acc', acc_val, epoch)
        log.train_writer.add_scalar('lr', lr, epoch)

def save_checkpoint(state, is_best: bool, filepath: str, filename: str):
    if (state['epoch']) % 10 == 0 or state['epoch'] == 1:
        os.makedirs(filepath, exist_ok=True)
        torch.save(state, os.path.join(filepath, filename))
        if rank == 0:
            log.printlog('Checkpoint saved!')
            if is_best:
                torch.save(state, os.path.join(filepath, 'model_best.pth'))
                log.printlog('Best model saved!')

def adjust_learning_rate_withWarmup(optimizer, epoch: int, args) -> float:
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    global rank, local_rank, log, device, scaler

    print(f'>>>>  Running on {device}  <<<<')

    model = Xmodel(embed_dim=3500, num_classes=args.num_classes)
    model.to(device)
    if rank == 0:
        log.printlog(model)

    trainset = ASEDataset(args.data_dir_train, args.atom_embed)
    valset = ASEDataset(args.data_dir_val, args.atom_embed)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)  # 启用未使用参数检测
    else:
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    criterion = FocalLoss(class_num=args.num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    # 早停止相关变量
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    patience = args.patience

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        loss_train, acc_train = run_one_epoch(model, train_loader, criterion, optimizer, epoch, mode='Train')
        loss_val, acc_val = run_one_epoch(model, val_loader, criterion, optimizer, epoch, mode='Eval')

        if rank == 0:
            print_log(epoch, loss_train, loss_val, acc_train, acc_val, optimizer.param_groups[0]['lr'])
            save_checkpoint({'epoch': epoch,
                             'model': model.module.state_dict() if distributed else model.state_dict(),
                             'optimizer': optimizer.state_dict()}, is_best=False,
                            filepath=f'{log.get_path()}/checkpoints/',
                            filename=f'checkpoint_{epoch:04d}.pth')

            # 检查验证集损失是否有改善
            if loss_val < best_loss:
                best_loss = loss_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # 检查是否需要早停止
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                early_stop = True
                break

        if early_stop:
            break

if __name__ == '__main__':
    rank, local_rank = 0, 0
    distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl")
            print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        
        distributed = True
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # 设置环境变量以获取详细调试信息
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--num_workers', default=16, type=int, metavar='N')
    parser.add_argument('--warmup_epochs', default=20, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--data_dir_train', nargs='+', default=['/data/cb_dataset/train.db'], type=str, help='List of training data directories, e.g., --data_dir_train= /path/to/train1.db /path/to/train2.db')
    parser.add_argument('--data_dir_val', nargs='+', default=['/data/cb_dataset/val.db'], type=str, help='List of validation data directories, e.g., --data_dir_val= /path/to/val.db')
    parser.add_argument('--atom_embed', type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--num_classes', default=100315, type=int, metavar='N')
    parser.add_argument('--patience', default=5, type=int, metavar='N', help='early stopping patience')

    args = parser.parse_args()

    if rank == 0:
        log = Logger(val=True)

    main()
    print('THE END')
