# -*- coding: utf-8 -*-
import pickle  # 用于保存匹配字典
from ase.db import connect
from ase import Atoms
import numpy as np
from tqdm import tqdm
import concurrent.futures
import traceback
from scipy.interpolate import interp1d
import spglib

# ================= 辅助函数 =================
def prim2conv(prim_atom):
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

def upsample(rows):
    rows = np.array(rows, dtype=object)
    _, unique_indices = np.unique(rows[:, 0], return_index=True)
    rows = rows[unique_indices]

    if float(rows[0][0]) > 10:
        rows = np.insert(rows, 0, ['10', float(rows[0][1])], axis=0)

    if float(rows[-1][0]) < 80:
        rows = np.append(rows, [['80', float(rows[-1][1])]], axis=0)

    rowsData = np.array(rows, dtype=np.float32)
    x = rowsData[:, 0].astype(np.float32)
    y = rowsData[:, 1].astype(np.float32)
    f = interp1d(x, y, kind='slinear', fill_value="extrapolate")
    xnew = np.linspace(10, 80, 3500)
    ynew = f(xnew)

    return ynew

# =============== 数据库连接 ===============
rruff = connect('RRUFF.db')
mp = connect('UniqCryLabeled.db')
savedb = connect('rruff2mp.db')

rruff_NUM = rruff.count()
mp_NUM = mp.count()

# =============== 预加载 MP 数据 ===============
mp_data = []
for row in mp.select():
    try:
        atoms = row.toatoms()
        latt_consts, _, c_atom = prim2conv(atoms)
        elements = set(c_atom.get_chemical_symbols())
        mp_data.append({
            'mpid': row.mpid,
            'Label': row.Label,
            'latt_consts': latt_consts,
            'elements': elements,
        })
    except Exception:
        print(f"Error loading mp entry id={row.mpid}")
        traceback.print_exc()

# =============== 匹配记录 ===============
match_dict = {}

def write_match_info(rruffid, mpid, match_type):
    with open('matched_pairs.txt', 'a', encoding='utf-8') as f:
        f.write(f"Matched RRUFFID={rruffid} <--> MPID={mpid} ({match_type})\n")

# =============== 保存匹配结果 ===============
def save_match(rruff_atoms, angle, intensity, rruffid, mp_entry, match_type):
    int_int = upsample(np.column_stack((eval(angle), eval(intensity))))
    int_int = int_int / int_int.max() * 100


    n_dis =  ', '.join(map(str, np.linspace(10, 80, 3500)))
    n_int =  ', '.join(map(str, int_int))
    
    savedb.write(
        atoms=rruff_atoms,
        angle=n_dis,
        intensity=n_int,
        RRUFFID=rruffid,
        mpid=mp_entry['mpid'],
        Label=int(mp_entry['Label'])
    )

    match_dict[rruffid] = mp_entry['mpid']
    write_match_info(rruffid, mp_entry['mpid'], match_type)
    print(f"[Matched-{match_type}] RRUFFID={rruffid} <--> MPID={mp_entry['mpid']}")

# =============== 匹配函数 ===============
def process_rruff_id(rruff_id):
    try:
        rruff_row = rruff.get(rruff_id)
        rruff_atoms = rruff.get_atoms(rruff_id)
        rf_latt_consts = rruff_atoms.cell.cellpar()
        rruff_elements = set(rruff_atoms.get_chemical_symbols())

        angle = getattr(rruff_row, 'angle', None)
        intensity = getattr(rruff_row, 'intensity', None)
        rruffid = getattr(rruff_row, 'RRUFFID', rruff_id)

        if angle is None or intensity is None:
            print(f"[Warning] Missing angle or intensity: rruff_id={rruff_id}")
            return False

        def is_lattice_match(mp_latt_consts):
            return all(mp_latt_consts[i] * 0.95 <= rf_latt_consts[i] <= mp_latt_consts[i] * 1.05 for i in range(6))

        def st_is_lattice_match(mp_latt_consts):
            return all(mp_latt_consts[i] * 0.99 <= rf_latt_consts[i] <= mp_latt_consts[i] * 1.01 for i in range(6))

        # ---------- 第一轮：严格匹配 ----------
        for mp_entry in mp_data:
            if is_lattice_match(mp_entry['latt_consts']) and rruff_elements == mp_entry['elements']:
                save_match(rruff_atoms, angle, intensity, rruffid, mp_entry, match_type="strict")
                return True

        # ---------- 第二轮：放松匹配 ----------
        for mp_entry in mp_data:
            if st_is_lattice_match(mp_entry['latt_consts']):
                save_match(rruff_atoms, angle, intensity, rruffid, mp_entry, match_type="relaxed")
                return True

        return False

    except Exception as e:
        print(f"[Error] rruff_id={rruff_id}: {e}")
        traceback.print_exc()
        return False

# =============== 主函数 ===============
if __name__ == "__main__":
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for success in tqdm(executor.map(process_rruff_id, range(1, rruff_NUM + 1)), total=rruff_NUM):
            if success:
                success_count += 1

    print(f"\n[Summary] Matched entries: {success_count} / {rruff_NUM}")

    # 保存匹配字典
    with open('matched_dict.pkl', 'wb') as f:
        pickle.dump(match_dict, f)

    print("[Done] Matching dictionary saved as matched_dict.pkl")

