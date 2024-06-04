import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_


class Xmodel(nn.Module):

    def __init__(self, embed_dim=3500, nhead=5, num_encoder_layers=6, dim_feedforward=192,
                 dropout=0., activation="relu", num_classes=10):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.conv = ConvModule(drop_rate=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, dim_feedforward))


        # -------------encoder----------------
        sa_layer = CrossAttnLayer(embed_dim, nhead, dim_feedforward, dropout, activation)
        self.encoder = SelfAttnModule(sa_layer, num_encoder_layers,)
        # ------------------------------------

        self.norm_after = nn.LayerNorm(embed_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), num_classes)
        )

        self._reset_parameters()
        self.init_weights()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights(self):
        trunc_normal_(self.cls_token, std=.02)

        self.pos_embed.requires_grad = False

        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.array(range(self.pos_embed.shape[2])))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).T.unsqueeze(0))

    def bce_fineTune_init_weights(self):
        for p in self.conv.parameters():
            p.requires_grad = False

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.cls_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, elem):
        x = x[:, :3500]
        N = x.shape[0]

        # normalize to 0-1
        x = x / 100
        x = x.unsqueeze(1) # N*1*3500
        
        x = self.conv(x) # N*192*3500

        # flatten NxCxL to CxNxL
        x = x.permute(1, 0, 2).contiguous() # 192*N*3500
        print(x.shape)

        """
        cls_token = self.cls_token.expand(-1, N, -1)
        print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=0) # 193*N*3501
        print(x.shape)
        """


        pos_embed = self.pos_embed.permute(2, 0, 1).contiguous().repeat(1, N, 1) 
        print(pos_embed.shape) # 192*N*3500

        elem = elem.unsqueeze(1) # N*1*92
        elem = elem.permute(1, 0, 2).contiguous() # 1*N*92

        feats = self.encoder(x, pos_embed,elem)
        feats = self.norm_after(feats)
        print(feats.shape)
        logits = self.cls_head(feats[0])
        print(logits.shape)
     
        return logits


class ConvModule(nn.Module):
    def __init__(self, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=17, stride=1, padding=(17 - 1) // 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=33, stride=1, padding=(33 - 1) // 2)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=65, stride=1, padding=(65 - 1) // 2)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=129, stride=1, padding=(129 - 1) // 2)
        self.bn4 = nn.BatchNorm1d(32)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=257, stride=1, padding=(257 - 1) // 2)
        self.bn5 = nn.BatchNorm1d(32)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=513, stride=1, padding=(513 - 1) // 2)
        self.bn6 = nn.BatchNorm1d(32)
        self.act6 = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = Layer(64, 64, kernel_size=3, stride=2, downsample=True)
        # self.layer2 = Layer(64, 128, kernel_size=3, stride=2, downsample=True)
        # self.layer3 = Layer(256, 256, kernel_size=3, stride=2, downsample=True)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.act3(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.act4(x4)

        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        x5 = self.act5(x5)

        x6 = self.conv6(x)
        x6 = self.bn6(x6)
        x6 = self.act6(x6)



        #x = self.maxpool(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        # x = self.layer3(x)
        #x = self.maxpool2(x)
        return torch.cat((x1, x2, x3, x4, x5, x6), dim=1)


class SelfAttnModule(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
   

    def forward(self, src, pos,elem):
        output = src

        for layer in self.layers:
            output = layer(output, pos,elem)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CrossAttnLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Define element_map
        self.element_map = nn.Sequential(
            nn.Linear(92, 1024),  # Increase the dimensionality
            nn.ReLU(),
            nn.Linear(1024, 192 * 3500) 
        )

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos, elem):
        q = self.element_map(elem).view(192, -1, 3500) # 192, N, 3500
        k = v = with_pos_embed(src, pos)
        src2 = self.cross_attn(q, k, value=v)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class Layer(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample):
        super(Layer, self).__init__()
        self.block1 = BasicBlock(inchannel, outchannel, kernel_size=kernel_size, stride=stride, downsample=downsample)
        self.block2 = BasicBlock(outchannel, outchannel, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=2),
            nn.BatchNorm1d(outchannel)
        ) if downsample else None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out).astype(np.float32)  # (M, D/2)
    emb_cos = np.cos(out).astype(np.float32)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
