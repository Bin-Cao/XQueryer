import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.dataset_for_face import EXPDataset
from model.XQueryer import Xmodel
from torch.cuda.amp import autocast
import os
from ase.db import connect
from ase import Atoms
import json
from mp_api.client import MPRester
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.diffraction import xrd

def get_acc(cls, label):
    correct_cnt = (cls.argmax(1) == label.int()).sum().item()
    cls_acc = correct_cnt / cls.shape[0]
    return cls_acc, correct_cnt

def run_one_epoch(model, dataloader, device):
    model.eval()  
    for batch in dataloader:
        intensity = batch['intensity'].to(device)
        element = batch['element'].to(device)
        with torch.no_grad():
            with autocast():
                logits = model(intensity, element)
    return logits,batch['intensity'].squeeze().numpy()[:-1]


def XQueryer(api_key,Top_num=1,Device='cpu',Load_path='./pretrained/checkpoint.pth',XY_data_dir='./XY_data.csv', ElementsSystem=[], ):
    """
    Contributors : Bin Cao
    ================================================================
        Please feel free to open issues in the Github :
        https://github.com/Bin-Cao/XQueryer
        or
        contact Mr.Bin Cao (bcao686@connect.hkust-gz.edu.cn)
        in case of any problems/comments/suggestions in using the code.
    ==================================================================
    :param api_key, type=str, required=True,
        Your Materials Project API key. see here : https://next-gen.materialsproject.org/api

    :param Top_num, type=int, default=5,
        The number of top predicted materials to be retrieved.

    :param XY_data_dir, type=str, default='./XY_data.csv',
        The directory of the XY data.   

    :param Device, type=str, default='cuda:0',
        Which device to run the XQueryer, example: 'cuda:0', 'cpu'.

    :param Load_path, type=str, default='./pretrained/checkpoint.pth' 
    
    :param ElementsSystem, type=list, desfault=[]
        Specifies the elements to be included at least in the prediction, example: ['Fe'].

    
    """
    
    os.makedirs('infere', exist_ok=True)
    data_dir = ['./infere/infere.db']
    if os.path.exists(data_dir[0]):
        os.remove(data_dir[0])
    xy2db(XY_data_dir,data_dir,ElementsSystem)
    device = torch.device(Device if torch.cuda.is_available() else 'cpu')
    model = Xmodel(embed_dim=3500, num_classes=100315)
    model.load_state_dict(torch.load(Load_path, map_location=device)['model'])
    model.to(device)
    model.eval()
    print('Loaded model from {}'.format(Load_path))

    valset = EXPDataset(data_dir, True)
    val_loader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
    probs, exp_y = run_one_epoch(model, val_loader, device)
    os.remove(data_dir[0])

    docs_list = []
    for i in range(Top_num):
        sorted_indices = np.argsort(probs, axis=1)
        docs = RetrieveMP(sorted_indices[:, -(i+1)],api_key)
        docs_list.append(docs)
    

    for i, doc in enumerate(docs_list):
        print(f'Top {i+1} predicted materials:')
        print(f'{i+1}. {doc[0].material_id} - {doc[0].formula_pretty} - {doc[0].chemsys}')
        print(doc[0].structure,'\n' )
        plt.figure(figsize=(8, 6)) 
        exp_x = np.arange(10,80,0.02)
        plt.plot(exp_x, exp_y * 100/exp_y.max(),  color='royalblue', linewidth=1) 
        plt.xlabel('2θ (degrees)', fontsize=14, weight='bold')
        plt.ylabel('Intensity (a.u.)', fontsize=14, weight='bold')
        mu_array,_Ints = get_diff(doc[0].structure)
        for i, (mu, intensity) in enumerate(zip(mu_array, _Ints)):
            plt.vlines(mu, ymin=0, ymax=intensity, color='k', linestyle='dashed', )
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
        plt.tight_layout() 
        plt.show()

    return docs_list


def get_diff(structure):
    # atom defined in ASE database
    calculator = xrd.XRDCalculator()
    pattern = calculator.get_pattern(structure, two_theta_range=(10, 80))
    # Due to the limitations of the package, a slight approximation is introduced here. 
    # The peak position is determined according to the set precision
    return pattern.x, pattern.y

def RetrieveMP(res,api_key):
    # Load the dictionary from the JSON file
    with open('./model/entries_dict.json', 'r') as file:
        entries_dict = json.load(file)
    row = entries_dict[f'{int(res)}']
    
    print('The MP link is : ', 'https://next-gen.materialsproject.org/materials/' + row['value'][:-4])
    with MPRester(api_key=api_key) as mpr:
        # retrieve SummaryDocs for a list of materials
        docs = mpr.summary.search(material_ids=[row['value'][:-4]])
    return docs

def xy2db(XY_data_dir, data_dir, ElementsSystem=None,):

    XY = pd.read_csv(XY_data_dir, skiprows=1,header=None)
    # XY = df.map(lambda x: np.format_float_positional(x, precision=4, unique=False, fractional=False) if isinstance(x, float) else x)
    if ElementsSystem:
        _atom = Atoms(ElementsSystem)
    else:
        _atom = Atoms(['H']) 
    databs = connect(data_dir[0])

    # Write to the database
    _x = str(XY.iloc[:, 0].tolist())
    _y = str(XY.iloc[:, 1].tolist())
    databs.write(
        atoms=_atom,  
        angle= _x,  # Lattice distances
        intensity= _y,  # Intensities、
        _element=str(ElementsSystem)  # Element list
    )
    
