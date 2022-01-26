from ..egt_molgraph import EGT_MOL

class EGT_PCQM4MV2(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)
        
    def output_block(self, g):
        h = super().output_block(g)
        return h.squeeze(-1)
