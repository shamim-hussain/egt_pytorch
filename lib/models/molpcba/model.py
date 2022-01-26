from ..egt_molgraph import EGT_MOL

class EGT_MOLPCBA(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=128, **kwargs)