import torch
from torch import nn
import torch.nn.functional as F
from .egt import EGT
from .egt_layers import VirtualNodes


NODE_FEATURES_OFFSET = 128
NUM_NODE_FEATURES = 9
EDGE_FEATURES_OFFSET = 8
NUM_EDGE_FEATURES = 3


class EGT_MOL(EGT):
    def __init__(self,
                 upto_hop          = 16,
                 mlp_ratios        = [1., 1.],
                 num_virtual_nodes = 0,
                 svd_encodings     = 0,
                 output_dim        = 1,
                 **kwargs):
        super().__init__(node_ended=True, **kwargs)
        
        self.upto_hop          = upto_hop
        self.mlp_ratios        = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_encodings     = svd_encodings
        self.output_dim        = output_dim
        
        self.nodef_embed = nn.Embedding(NUM_NODE_FEATURES*NODE_FEATURES_OFFSET+1,
                                        self.node_width, padding_idx=0)
        if self.svd_encodings:
            self.svd_embed = nn.Linear(self.svd_encodings*2, self.node_width)
        
        self.dist_embed = nn.Embedding(self.upto_hop+2, self.edge_width)
        self.featm_embed = nn.Embedding(NUM_EDGE_FEATURES*EDGE_FEATURES_OFFSET+1,
                                        self.edge_width, padding_idx=0)
        
        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(self.node_width, self.edge_width, 
                                         self.num_virtual_nodes)
        
        self.final_ln_h = nn.LayerNorm(self.node_width)
        mlp_dims = [self.node_width * max(self.num_virtual_nodes, 1)]\
                    +[round(self.node_width*r) for r in self.mlp_ratios]\
                        +[self.output_dim]
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i],mlp_dims[i+1])
                                         for i in range(len(mlp_dims)-1)])
        self.mlp_fn = getattr(F, self.activation)
    
    
    def input_block(self, inputs):
        g = super().input_block(inputs)
        nodef = g.node_features.long()              # (b,i,f)
        nodem = g.node_mask.float()                 # (b,i)
        
        dm0 = g.distance_matrix                     # (b,i,j)
        dm = dm0.long().clamp(max=self.upto_hop+1)  # (b,i,j)
        featm = g.feature_matrix.long()             # (b,i,j,f)
        
        h = self.nodef_embed(nodef).sum(dim=2)      # (b,i,w,h) -> (b,i,h)
        
        if self.svd_encodings:
            h = h + self.svd_embed(g.svd_encodings)
        
        e = self.dist_embed(dm)\
              + self.featm_embed(featm).sum(dim=3)  # (b,i,j,f,e) -> (b,i,j,e)
        
        g.mask = (nodem[:,:,None,None] * nodem[:,None,:,None] - 1)*1e9
        g.h, g.e = h, e
        
        if self.num_virtual_nodes > 0:
            g = self.vn_layer(g)
        return g
    
    def final_embedding(self, g):
        h = g.h
        h = self.final_ln_h(h)
        if self.num_virtual_nodes > 0:
            h = h[:,:self.num_virtual_nodes].reshape(h.shape[0],-1)
        else:
            nodem = g.node_mask.float().unsqueeze(dim=-1)
            h = (h*nodem).sum(dim=1)/(nodem.sum(dim=1)+1e-9)
        g.h = h
        return g
    
    def output_block(self, g):
        h = g.h
        h = self.mlp_layers[0](h)
        for layer in self.mlp_layers[1:]:
            h = layer(self.mlp_fn(h))
        return h


