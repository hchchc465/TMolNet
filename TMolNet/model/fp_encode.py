import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
import torch.nn as nn
import torch.nn.functional as F

def calculate_combined_fingerprint(smiles):
    """
    计算分子的组合指纹，包含MACCS、PubChem和Pharmacophore ErG三种指纹

    参数:
        smiles (str): 分子的SMILES表示

    返回:
        np.array: 拼接后的指纹向量，长度为167 (MACCS) + 881 (PubChem) + 200 (ErG) = 1248
    """
    try:
        # 计算MACCS指纹 (167位)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            maccs = np.zeros(167, dtype=int)
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            maccs = np.array(list(fp.ToBitString()), dtype=int)

        # 计算PubChem指纹 (881位)
        if mol is None:
            pubchem = np.zeros(881, dtype=int)
        else:
            # 使用原子对指纹模拟PubChem
            fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881)
            pubchem = np.array(list(fp.ToBitString()), dtype=int)

        # 计算Pharmacophore ErG指纹 (200位)
        if mol is None:
            pharmacophore = np.zeros(200, dtype=int)
        else:
            # 使用ErG特征模拟药效团指纹
            fp = AllChem.GetErGFingerprint(mol)
            pharmacophore = np.zeros(200, dtype=int)
            for idx in fp.GetNonzeroElements():
                if idx < 200:
                    pharmacophore[idx] = 1

        # 拼接三种指纹
        return np.concatenate([maccs, pubchem, pharmacophore])

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {str(e)}")
        return np.zeros(1248, dtype=int)


# ----------------------
# 混合指纹编码器
# ----------------------

class FingerprintEncoder(nn.Module):
    def __init__(self, input_dim=1248, hidden_dims=[512, 256], output_dim=256, dropout=0.2):
        """
        参数:
            input_dim: 输入维度，默认是 1248（三种指纹拼接）
            hidden_dims: MLP 隐藏层维度列表
            output_dim: 输出特征维度，默认为 256
            dropout: Dropout 比例，默认 0.2
        """
        super(FingerprintEncoder, self).__init__()

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h

        # 最后一层输出为 256 维
        layers.append(nn.Linear(last_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch_size, 1248]
        return: [batch_size, 256]
        """
        return self.mlp(x)






