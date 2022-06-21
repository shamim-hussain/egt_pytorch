
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Graph(dict):
    def __dir__(self):
        return super().__dir__() + list(self.keys())
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError('No such attribute: '+key)
        
    def __setattr__(self, key, value):
        self[key]=value
        
    def copy(self):
        return self.__class__(self)



class EGT_Layer(nn.Module):
    @staticmethod
    @torch.jit.script
    def _egt(scale_dot: bool,
             scale_degree: bool,
             num_heads: int,
             dot_dim: int,
             clip_logits_min: float,
             clip_logits_max: float,
             attn_dropout: float,
             attn_maskout: float,
             training: bool,
             num_vns: int,
             QKV: torch.Tensor,
             G: torch.Tensor,
             E: torch.Tensor,
             mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0],shp[1],-1,num_heads).split(dot_dim,dim=2)
        
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        
        if mask is None:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G)#+rmask
                A_tild = F.softmax(H_hat+rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G)
                A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G+mask)
                A_tild = F.softmax(H_hat+mask+rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G+mask)
                A_tild = F.softmax(H_hat+mask, dim=2) * gates
        
        if attn_dropout > 0:
            A_tild = F.dropout(A_tild, p=attn_dropout, training=training)
        
        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)
        
        if scale_degree:
            degrees = torch.sum(gates,dim=2,keepdim=True)
            degree_scalers = torch.log(1+degrees)
            degree_scalers[:,:num_vns] = 1.
            V_att = V_att * degree_scalers
        
        V_att = V_att.reshape(shp[0],shp[1],num_heads*dot_dim)
        return V_att, H_hat

    @staticmethod
    @torch.jit.script
    def _egt_edge(scale_dot: bool,
                  num_heads: int,
                  dot_dim: int,
                  clip_logits_min: float,
                  clip_logits_max: float,
                  QK: torch.Tensor,
                  E: torch.Tensor) -> torch.Tensor:
        shp = QK.shape
        Q, K = QK.view(shp[0],shp[1],-1,num_heads).split(dot_dim,dim=2)
        
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        return H_hat
    
    def __init__(self,
                 node_width                      ,
                 edge_width                      ,
                 num_heads                       ,
                 node_mha_dropout    = 0         ,
                 edge_mha_dropout    = 0         ,
                 node_ffn_dropout    = 0         ,
                 edge_ffn_dropout    = 0         ,
                 attn_dropout        = 0         ,
                 attn_maskout        = 0         ,
                 activation          = 'elu'     ,
                 clip_logits_value   = [-5,5]    ,
                 node_ffn_multiplier = 2.        ,
                 edge_ffn_multiplier = 2.        ,
                 scale_dot           = True      ,
                 scale_degree        = False     ,
                 node_update         = True      ,
                 edge_update         = True      ,
                 ):
        super().__init__()
        self.node_width          = node_width         
        self.edge_width          = edge_width          
        self.num_heads           = num_heads           
        self.node_mha_dropout    = node_mha_dropout        
        self.edge_mha_dropout    = edge_mha_dropout        
        self.node_ffn_dropout    = node_ffn_dropout        
        self.edge_ffn_dropout    = edge_ffn_dropout        
        self.attn_dropout        = attn_dropout
        self.attn_maskout        = attn_maskout
        self.activation          = activation          
        self.clip_logits_value   = clip_logits_value   
        self.node_ffn_multiplier = node_ffn_multiplier 
        self.edge_ffn_multiplier = edge_ffn_multiplier 
        self.scale_dot           = scale_dot
        self.scale_degree        = scale_degree        
        self.node_update         = node_update         
        self.edge_update         = edge_update        
        
        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width//self.num_heads
        
        self.mha_ln_h   = nn.LayerNorm(self.node_width)
        self.mha_ln_e   = nn.LayerNorm(self.edge_width)
        self.lin_E      = nn.Linear(self.edge_width, self.num_heads)
        if self.node_update:
            self.lin_QKV    = nn.Linear(self.node_width, self.node_width*3)
            self.lin_G      = nn.Linear(self.edge_width, self.num_heads)
        else:
            self.lin_QKV    = nn.Linear(self.node_width, self.node_width*2)
        
        self.ffn_fn     = getattr(F, self.activation)
        if self.node_update:
            self.lin_O_h    = nn.Linear(self.node_width, self.node_width)
            if self.node_mha_dropout > 0:
                self.mha_drp_h  = nn.Dropout(self.node_mha_dropout)
            
            node_inner_dim  = round(self.node_width*self.node_ffn_multiplier)
            self.ffn_ln_h   = nn.LayerNorm(self.node_width)
            self.lin_W_h_1  = nn.Linear(self.node_width, node_inner_dim)
            self.lin_W_h_2  = nn.Linear(node_inner_dim, self.node_width)
            if self.node_ffn_dropout > 0:
                self.ffn_drp_h  = nn.Dropout(self.node_ffn_dropout)
        
        if self.edge_update:
            self.lin_O_e    = nn.Linear(self.num_heads, self.edge_width)
            if self.edge_mha_dropout > 0:
                self.mha_drp_e  = nn.Dropout(self.edge_mha_dropout)
        
            edge_inner_dim  = round(self.edge_width*self.edge_ffn_multiplier)
            self.ffn_ln_e   = nn.LayerNorm(self.edge_width)
            self.lin_W_e_1  = nn.Linear(self.edge_width, edge_inner_dim)
            self.lin_W_e_2  = nn.Linear(edge_inner_dim, self.edge_width)
            if self.edge_ffn_dropout > 0:
                self.ffn_drp_e  = nn.Dropout(self.edge_ffn_dropout)
    
    def forward(self, g):
        h, e = g.h, g.e
        mask = g.mask
        
        h_r1 = h
        e_r1 = e
        
        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        
        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)
        
        if self.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(self.scale_dot,
                                     self.scale_degree,
                                     self.num_heads,
                                     self.dot_dim,
                                     self.clip_logits_value[0],
                                     self.clip_logits_value[1],
                                     self.attn_dropout,
                                     self.attn_maskout,
                                     self.training,
                                     0 if 'num_vns' not in g else g.num_vns,
                                     QKV,
                                     G, E, mask)
            
            h = self.lin_O_h(V_att)
            if self.node_mha_dropout > 0:
                h = self.mha_drp_h(h)
            h.add_(h_r1)
            
            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            h.add_(h_r2)
        else:
            H_hat = self._egt_edge(self.scale_dot,
                                   self.num_heads,
                                   self.dot_dim,
                                   self.clip_logits_value[0],
                                   self.clip_logits_value[1],
                                   QKV, E)
        
        
        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            e.add_(e_r1)
            
            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            e.add_(e_r2)
        
        g = g.copy()
        g.h, g.e = h, e
        return g
    
    def __repr__(self):
        rep = super().__repr__()
        rep = (rep + ' ('
                   + f'num_heads: {self.num_heads},'
                   + f'activation: {self.activation},'
                   + f'attn_maskout: {self.attn_maskout},'
                   + f'attn_dropout: {self.attn_dropout}'
                   +')')
        return rep



class VirtualNodes(nn.Module):
    def __init__(self, node_width, edge_width, num_virtual_nodes = 1):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_virtual_nodes = num_virtual_nodes

        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.node_width))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.edge_width))
        nn.init.normal_(self.vn_node_embeddings)
        nn.init.normal_(self.vn_edge_embeddings)
    
    def forward(self, g):
        h, e = g.h, g.e
        mask = g.mask
        
        node_emb = self.vn_node_embeddings.unsqueeze(0).expand(h.shape[0], -1, -1)
        h = torch.cat([node_emb, h], dim=1)
        
        e_shape = e.shape
        edge_emb_row = self.vn_edge_embeddings.unsqueeze(1)
        edge_emb_col = self.vn_edge_embeddings.unsqueeze(0)
        edge_emb_box = 0.5 * (edge_emb_row + edge_emb_col)
        
        edge_emb_row = edge_emb_row.unsqueeze(0).expand(e_shape[0], -1, e_shape[2], -1)
        edge_emb_col = edge_emb_col.unsqueeze(0).expand(e_shape[0], e_shape[1], -1, -1)
        edge_emb_box = edge_emb_box.unsqueeze(0).expand(e_shape[0], -1, -1, -1)
        
        e = torch.cat([edge_emb_row, e], dim=1)
        e_col_box = torch.cat([edge_emb_box, edge_emb_col], dim=1)
        e = torch.cat([e_col_box, e], dim=2)
        
        g = g.copy()
        g.h, g.e = h, e
        
        g.num_vns = self.num_virtual_nodes
        
        if mask is not None:
            g.mask = F.pad(mask, (0,0, self.num_virtual_nodes,0, self.num_virtual_nodes,0), 
                           mode='constant', value=0)
        return g

