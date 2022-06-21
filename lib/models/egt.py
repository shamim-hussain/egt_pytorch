
import torch
from torch import nn
import torch.nn.functional as F
from .egt_layers import EGT_Layer, Graph


class EGT_Base(nn.Module):
    def __init__(self,
                 node_width          = 128       ,
                 edge_width          = 32        ,
                 num_heads           = 8         ,
                 model_height        = 4         ,
                 node_mha_dropout    = 0.        ,
                 node_ffn_dropout    = 0.        ,
                 edge_mha_dropout    = 0.        ,
                 edge_ffn_dropout    = 0.        ,
                 attn_dropout        = 0.        ,
                 attn_maskout        = 0.        ,
                 activation          = 'elu'     ,
                 clip_logits_value   = [-5,5]    ,
                 node_ffn_multiplier = 2.        ,
                 edge_ffn_multiplier = 2.        ,
                 scale_dot           = True      ,
                 scale_degree        = False     ,
                 node_ended          = False     ,
                 edge_ended          = False     ,
                 egt_simple          = False     ,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.node_width          = node_width      
        self.edge_width          = edge_width       
        self.num_heads           = num_heads        
        self.model_height        = model_height       
        self.node_mha_dropout    = node_mha_dropout
        self.node_ffn_dropout    = node_ffn_dropout
        self.edge_mha_dropout    = edge_mha_dropout     
        self.edge_ffn_dropout    = edge_ffn_dropout     
        self.attn_dropout        = attn_dropout     
        self.attn_maskout        = attn_maskout
        self.activation          = activation       
        self.clip_logits_value   = clip_logits_value
        self.node_ffn_multiplier = node_ffn_multiplier 
        self.edge_ffn_multiplier = edge_ffn_multiplier 
        self.scale_dot           = scale_dot
        self.scale_degree        = scale_degree        
        self.node_ended          = node_ended          
        self.edge_ended          = edge_ended          
        self.egt_simple          = egt_simple          
        
        self.layer_common_kwargs = dict(
             node_width          = self.node_width            ,
             edge_width          = self.edge_width            ,
             num_heads           = self.num_heads             ,
             node_mha_dropout    = self.node_mha_dropout      ,
             node_ffn_dropout    = self.node_ffn_dropout      ,
             edge_mha_dropout    = self.edge_mha_dropout      ,
             edge_ffn_dropout    = self.edge_ffn_dropout      ,
             attn_dropout        = self.attn_dropout          ,
             attn_maskout        = self.attn_maskout          ,
             activation          = self.activation            ,
             clip_logits_value   = self.clip_logits_value     ,
             scale_dot           = self.scale_dot             ,
             scale_degree        = self.scale_degree          ,
             node_ffn_multiplier = self.node_ffn_multiplier   ,
             edge_ffn_multiplier = self.edge_ffn_multiplier   ,
        )
        
    def input_block(self, inputs):
        return Graph(inputs)
    
    def final_embedding(self, g):
        raise NotImplementedError
    
    def output_block(self, g):
        raise NotImplementedError
    
    def forward(self, inputs):
        raise NotImplementedError



    
class EGT(EGT_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        
        self.EGT_layers = nn.ModuleList([EGT_Layer(**self.layer_common_kwargs, 
                                                   edge_update=(not self.egt_simple))
                                         for _ in range(self.model_height-1)])
    
        if (not self.node_ended) and (not self.edge_ended):
            pass
        elif not self.node_ended:
            self.EGT_layers.append(EGT_Layer(**self.layer_common_kwargs, node_update = False))
        elif not self.edge_ended:
            self.EGT_layers.append(EGT_Layer(**self.layer_common_kwargs, edge_update = False))
        else:
            self.EGT_layers.append(EGT_Layer(**self.layer_common_kwargs))
     
    def forward(self, inputs):
        g = self.input_block(inputs)
        
        for layer in self.EGT_layers:
            g = layer(g)
        
        g = self.final_embedding(g)
        
        outputs = self.output_block(g)
        return outputs

