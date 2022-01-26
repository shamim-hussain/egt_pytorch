import torch
import numpy as np
import torch.nn.functional as F

from lib.training.training import cached_property
from ..egt_mol_training import EGT_MOL_Training

from lib.models.pcqm4mv2 import EGT_PCQM4MV2
from lib.data.pcqm4mv2 import PCQM4Mv2StructuralSVDGraphDataset

class PCQM4MV2_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name = 'pcqm4mv2',
            dataset_path = 'cache_data/PCQM4MV2',
            predict_on   = ['val','test'],
            evaluate_on  = ['val','test'],
            state_file   = None,
        )
        return config_dict
    
    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, PCQM4Mv2StructuralSVDGraphDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_PCQM4MV2
    
    def calculate_loss(self, outputs, inputs):
        return F.l1_loss(outputs, inputs['target'])
    
    @cached_property
    def evaluator(self):
        from ogb.lsc.pcqm4mv2 import PCQM4Mv2Evaluator
        evaluator = PCQM4Mv2Evaluator()
        return evaluator
    
    def prediction_step(self, batch):
        return dict(
            predictions = self.model(batch),
            targets     = batch['target'],
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if dataset_name == 'test':
            self.evaluator.save_test_submission(
                input_dict = {'y_pred': predictions['predictions']},
                dir_path = self.config.predictions_path,
                mode = 'test-dev',
            )
            print(f'Saved final test-dev predictions to {self.config.predictions_path}')
            return {'mae': np.nan}
        
        print(f'Evaluating on {dataset_name}')
        input_dict = {"y_true": predictions['targets'], 
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results

SCHEME = PCQM4MV2_Training
