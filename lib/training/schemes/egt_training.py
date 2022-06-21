
from lib.training.training import TrainingBase, cached_property, CollatedBatch
from lib.training.testing import TestingBase
from contextlib import nullcontext
from lib.training.training_mixins import SaveModel, VerboseLR
from lib.utils.dotdict import HDict
import torch
from lib.data.graph_dataset import graphdata_collate

class EGTTraining(TestingBase,TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name          = 'egt',
            cache_dir           = 'cache_data',
            dataset_name        = 'unnamed_dataset',
            dataset_path        = HDict.L('c:f"{c.cache_dir}/{c.dataset_name.upper()}"'),
            save_path           = HDict.L('c:path.join(f"models/{c.dataset_name.lower()}",c.model_name)'),
            model_height        = 4,
            node_width          = 64,
            edge_width          = 64,
            num_heads           = 8,
            node_dropout        = 0.,
            edge_dropout        = 0.,
            node_ffn_dropout    = HDict.L('c:c.node_dropout'),
            edge_ffn_dropout    = HDict.L('c:c.edge_dropout'),
            attn_dropout        = 0.,
            attn_maskout        = 0.,
            activation          = 'elu',
            clip_logits_value   = [-5,5],
            scale_degree        = True,
            node_ffn_multiplier = 1.,
            edge_ffn_multiplier = 1.,
            allocate_max_batch  = True,
            scale_dot_product   = True,
            egt_simple          = False,
        )
        return config
    
    
    def get_dataset_config(self):
        config = self.config
        dataset_config = dict(
            dataset_path = config.dataset_path,
            cache_dir    = config.cache_dir,
        )
        return dataset_config, None
    
    def get_model_config(self):
        config = self.config
        model_config = dict(
            model_height        = config.model_height         ,
            node_width          = config.node_width           ,
            edge_width          = config.edge_width           ,
            num_heads           = config.num_heads            ,
            node_mha_dropout    = config.node_dropout         ,
            edge_mha_dropout    = config.edge_dropout         ,
            node_ffn_dropout    = config.node_ffn_dropout     ,
            edge_ffn_dropout    = config.edge_ffn_dropout     ,
            attn_dropout        = config.attn_dropout         ,
            attn_maskout        = config.attn_maskout         ,
            activation          = config.activation           ,
            clip_logits_value   = config.clip_logits_value    ,
            scale_degree        = config.scale_degree         ,
            node_ffn_multiplier = config.node_ffn_multiplier  ,
            edge_ffn_multiplier = config.edge_ffn_multiplier  ,
            scale_dot           = config.scale_dot_product    ,
            egt_simple          = config.egt_simple           ,
        )
        return model_config, None
    
    def _cache_dataset(self, dataset):
        if self.is_main_rank:
            dataset.cache()
        self.distributed_barrier()
        if not self.is_main_rank:
            dataset.cache(verbose=0)
    
    def _get_dataset(self, split):
        dataset_config, dataset_class = self.get_dataset_config()
        if dataset_class is None:
            raise NotImplementedError('Dataset class not specified')
        dataset = dataset_class(**dataset_config, split=split)
        self._cache_dataset(dataset)
        return dataset

    @cached_property
    def train_dataset(self):
        return self._get_dataset('training')
    @cached_property
    def val_dataset(self):
        return self._get_dataset('validation')
    @cached_property
    def test_dataset(self):
        return self._get_dataset('test')
    
    @property
    def collate_fn(self):
        return graphdata_collate
    
    @cached_property
    def base_model(self):
        model_config, model_class = self.get_model_config()
        if model_class is None:
            raise NotImplementedError
        model = model_class(**model_config).cuda()
        return model
    
    def prepare_for_training(self):
        # cache datasets in same order on all ranks
        if self.is_distributed:
            self.train_dataset
            self.val_dataset
        super().prepare_for_training()
        
        # GPU memory cache for biggest batch
        if self.config.allocate_max_batch:
            if self.is_main_rank: print('Allocating cache for max batch size...', flush=True)
            torch.cuda.empty_cache()
            self.model.train()
            max_batch = self.train_dataset.max_batch(self.config.batch_size, self.collate_fn)
            max_batch = self.preprocess_batch(max_batch)
            
            outputs = self.model(max_batch)
            loss = self.calculate_loss(outputs=outputs, inputs=max_batch)
            loss.backward()
            
            for param in self.model.parameters():
                param.grad = None
    
    def initialize_losses(self, logs, training):
        self._total_loss = 0.
        self._total_samples = 0.
    
    def update_losses(self, i, loss, inputs, logs, training):
        if not isinstance(inputs, CollatedBatch):
            step_samples = float(inputs['num_nodes'].shape[0])
        else:
            step_samples = float(sum(i['num_nodes'].shape[0] for i in inputs))
        if not self.is_distributed:
            step_loss = loss.item() * step_samples
        else:
            step_samples = torch.tensor(step_samples, device=loss.device,
                                        dtype=loss.dtype)
            
            if training:
                loss = loss.detach()
            step_loss = loss * step_samples
            
            torch.distributed.all_reduce(step_loss)
            torch.distributed.all_reduce(step_samples)
            
            step_loss = step_loss.item()
            step_samples = step_samples.item()
            
        self._total_loss += step_loss
        self._total_samples += step_samples
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/self._total_samples)

