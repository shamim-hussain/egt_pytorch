import torch
import torch.nn.functional as F

from lib.training.training import cached_property
from ..egt_mol_training import EGT_MOL_Training

from lib.models.molpcba import EGT_MOLPCBA
from lib.data.molpcba import MOLPCBAStructuralSVDGraphDataset

class MOLPCBA_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name    = 'molpcba',
            dataset_path    = 'cache_data/MOLPCBA',
            evaluation_type = 'prediction',
            predict_on      = ['test'],
            state_file      = None,
        )
        return config_dict
            
    
    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, MOLPCBAStructuralSVDGraphDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_MOLPCBA
    
    def calculate_masked_loss(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        targets_mask = (targets == targets)
        loss = F.binary_cross_entropy_with_logits(outputs[targets_mask],
                                                  targets[targets_mask])
        return loss
    
    def calculate_loss(self, outputs, inputs):
        return self.calculate_masked_loss(outputs, inputs['target'])
    
    @cached_property
    def evaluator(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name = "ogbg-molpcba")
        return evaluator
    
    def prediction_step(self, batch):
        return dict(
            predictions = torch.sigmoid(self.model(batch)),
            targets     = batch['target'],
        )
        
    def evaluate_predictions(self, predictions):
        input_dict = {"y_true": predictions['targets'], 
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)
        
        xent = self.calculate_masked_loss(torch.from_numpy(predictions['predictions']),
                                          torch.from_numpy(predictions['targets'])).item()
        results['xent'] = xent
        
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions)
        return results
        

SCHEME = MOLPCBA_Training
