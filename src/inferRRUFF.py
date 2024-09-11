import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.dataset import EXPDataset
from model.XQueryer import Xmodel
from torch.cuda.amp import autocast
import ase
from ase import Atoms
import spglib



def get_acc(cls, rruff_id):
    """
    Calculate the accuracy of the classification model.

    Parameters:
    - cls: Predictions from the classification model, shape (N, C), 
           where N is the number of samples and C is the number of classes.
    - rruff_id: Material IDs from the RRUFF database, shape (N,).

    Returns:
    - cls_acc: The accuracy of the classification model.
    - correct_cnt: The number of matching material pairs.
    """
    # MP database material IDs predicted by the model, starting from 1
    # prediction lable = MP ID - 1
    mp_id = cls.argmax(1) + 1  # N x 1
    rf_id = rruff_id.int()     # N x 1

    # Calculate the number of matching material pairs
    correct_cnt = check_match_num(mp_id, rf_id)
    cls_acc = correct_cnt / cls.shape[0]

    return cls_acc, correct_cnt

def run_one_epoch(model, dataloader, device):
    model.eval()


    correct_cnt, total_cnt = 0, 0
    pbar = tqdm(total=len(dataloader.dataset), desc='Evaluating... ', unit='data')
    iters = len(dataloader)
   

    for batch in dataloader:
        intensity = batch['intensity'].to(device)
        rruff_id = batch['id'].to(device)
        element = batch['element'].to(device)

        with torch.no_grad():
            with autocast():
                logits = model(intensity, element)

        pbar.update(len(intensity))

        _, correct = get_acc(logits, rruff_id)

        correct_cnt += correct
        total_cnt += len(intensity)

    pbar.close()
    return correct_cnt, total_cnt


def prim2conv(prim_atom):
    """
    Convert a primitive cell to a conventional cell.

    Parameters:
        prim_atom (Atoms): The primitive atom defined in the atomic simulation unit (asu).

    Returns:
        tuple: Lattice constants, conventional lattice cell matrix in Cartesian coordinates, Atoms attribute
    """
    lattice = prim_atom.get_cell()
    positions = prim_atom.get_scaled_positions()
    numbers = prim_atom.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    conventional_cell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True)
    conv_lattice, conv_positions, conv_numbers = conventional_cell
    conventional_atoms = Atoms(cell=conv_lattice, scaled_positions=conv_positions, numbers=conv_numbers, pbc=True)
    lc = conventional_atoms.cell.cellpar()
    lmtx = conventional_atoms.get_cell()[:]
    return lc, lmtx, conventional_atoms


def check_match_num(mp_id, rf_id):
    """
    Compare materials from the Materials Project (MP) and RRUFF databases,
    counting the number of matching material pairs.

    Parameters:
    - mp_id: List of material IDs from the MP database.
    - rf_id: List of material IDs from the RRUFF database.

    Returns:
    - cnt: The number of matching material pairs.
    """
    cnt = 0

    # Iterate over the material IDs from both databases
    for i in range(len(mp_id)):
        _mpid = int(mp_id[i])
        _rfid = int(rf_id[i])
        
        # Retrieve atomic information from the RRUFF database
        rruff_atoms = ase.db.connect(args.data_dir[0]).get_atoms(id=_rfid)
        rf_latt_consts, _, rf_c_atom = prim2conv(rruff_atoms)
        #rruff_latt_consts = rruff_atoms.cell.cellpar()
        rruff_element = set(rruff_atoms.get_chemical_symbols())

        # Retrieve atomic information from the MP database
        mp_atoms = ase.db.connect(args.mp_dir[0]).get_atoms(id=_mpid)
        mp_latt_consts, _, mp_c_atom = prim2conv(mp_atoms)
        #mp_positions = mp_c_atom.get_scaled_positions()
        mp_element = set(mp_c_atom.get_chemical_symbols())

        try:
            # Check if the number of atoms, lattice constants, and elemental composition match
            if (
                
                all(mp_latt_consts[i] * 0.95 <= rf_latt_consts[i] <= mp_latt_consts[i] * 1.05 for i in range(6))
                and rruff_element == mp_element
            ):
                cnt += 1
        except Exception as e:
            print(f"Error processing IDs {_mpid} and {_rfid}: {e}")
            # cnt += 1
            pass

    return cnt



def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = Xmodel(embed_dim=3500, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.load_path, map_location=device)['model'])
    model.to(device)
    model.eval()
    print('Loaded model from {}'.format(args.load_path))

    valset = EXPDataset(args.data_dir, args.atom_embed)
    val_loader = DataLoader(valset, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    correct_cnt, total_cnt = run_one_epoch(model, val_loader, device)
    print(f"Accuracy: {round(correct_cnt / total_cnt * 100, 2)}%  ({correct_cnt}/{total_cnt})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, choices=['cuda:0', 'cpu'])
    parser.add_argument('--data_dir', default=['/data/cb_dataset/RRUFF.db'], type=str)
    parser.add_argument('--mp_dir', default=['/data/cb_dataset/mpdata.db'], type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--atom_embed', default=True, type=bool)
    parser.add_argument('--load_path', default='/home/cb/XRDS/XQueryer/output/2024-08-09_1444/checkpoints/checkpoint_0010.pth', type=str,
                        help='Path to load pretrained single-phase identification model')
    parser.add_argument('--num_classes', default=100315, type=int)

    args = parser.parse_args()
    main()
    print('THE END')


