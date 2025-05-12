import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.conversions import rodrigues_batch, quaternion_to_matrix
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model) -> (seq_len, batch, d_model) if batch_first=False
        self.register_buffer('pe', pe)

    '''
    def forward(self, x):
        # x: (seq_len, batch_size, d_model) if batch_first=False for transformer
        # or (batch_size, seq_len, d_model) if batch_first=True
        x = x + self.pe[:x.size(0 if not self.dropout.training or not isinstance(x, nn.Dropout) else 1), :] # Adjust for batch_first
        return self.dropout(x)
    '''
    
    def forward(self, x):
        slice_dim_index = 0 if not self.dropout.training or not isinstance(x, nn.Dropout) else 1
        slice_len = x.size(slice_dim_index)
        pe_slice = self.pe[:slice_len, :]
        
        x = x + pe_slice 

        return self.dropout(x)
