import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from contextlib import nullcontext

import yaml
from yaml import SafeLoader as yaml_Loader, SafeDumper as yaml_Dumper
import os,sys

from tqdm import tqdm

from lib.utils.dotdict import HDict
HDict.L.update_globals({'path':os.path})

def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:  # check for multiline string
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def read_config_from_file(config_file):
    with open(config_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml_Loader)

def save_config_to_file(config, config_file):
    with open(config_file, 'w') as fp:
        return yaml.dump(config, fp, sort_keys=False, Dumper=yaml_Dumper)


class StopTrainingException(Exception):
    pass

class CollatedBatch(list):
    pass

class DistributedTestDataSampler(Sampler):
    def __init__(self, data_source, batch_size, rank, world_size):
        data_len = len(data_source)
        all_indices = np.arange(data_len, dtype=int)
        split_indices = np.array_split(all_indices, world_size)
        
        num_batches = (len(split_indices[0]) + batch_size -1) // batch_size
        self.batch_indices = [i.tolist() for i in np.array_split(split_indices[rank],
                                                                 num_batches)]
    
    def __iter__(self):
        return iter(self.batch_indices)
    
    def __len__(self):
        return len(self.batch_indices)



def cached_property(func):
    atrribute_name = f'_{func.__name__}'
    def _wrapper(self):
        try:
            return getattr(self, atrribute_name)
        except AttributeError:
            val = func(self)
            self.__dict__[atrribute_name] = val
            return val
    return property(_wrapper)


class TrainingBase:
    def __init__(self, config=None, ddp_rank=0, ddp_world_size=1):
        self.config_input = config
        self.config = self.get_default_config()
        if config is not None:
            for k in config.keys():
                if not k in self.config:
                    raise KeyError(f'Unknown config "{k}"')
            self.config.update(config)
        
        self.state = self.get_default_state()
        
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_distributed = (self.ddp_world_size > 1)
        self.is_main_rank = (self.ddp_rank == 0)


    @cached_property
    def train_dataset(self):
        raise NotImplementedError
    
    @cached_property
    def val_dataset(self):
        raise NotImplementedError
    
    @cached_property
    def collate_fn(self):
        return None

    @cached_property
    def train_sampler(self):
        return  torch.utils.data.DistributedSampler(self.train_dataset,
                                                    shuffle=True)
    
    @cached_property
    def train_dataloader(self):
        common_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
                multiprocessing_context=self.config.dataloader_mp_context,
            )
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs, shuffle=True,
                                    drop_last=False)
        else:
            dataloader = DataLoader(**common_kwargs, 
                                    sampler=self.train_sampler)
        return dataloader
    
    @cached_property
    def val_dataloader(self):
        common_kwargs = dict(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
                multiprocessing_context=self.config.dataloader_mp_context,
            )
        prediction_batch_size = self.config.batch_size*self.config.prediction_bmult
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs, 
                                    batch_size=prediction_batch_size,
                                    shuffle=False, drop_last=False)
        else:
            sampler = DistributedTestDataSampler(data_source=self.val_dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(**common_kwargs, batch_sampler=sampler)
        return dataloader

    @cached_property
    def base_model(self):
        raise NotImplementedError
    
    @cached_property
    def model(self):
        model = self.base_model
        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[self.ddp_rank],
                                                              output_device=self.ddp_rank)
        return model
    
    @cached_property
    def optimizer(self):
        config = self.config
        optimizer_class = getattr(torch.optim, config.optimizer)
        optimizer = optimizer_class(self.model.parameters(),
                                            lr=config.max_lr, 
                                            **config.optimizer_params)
        return optimizer

    def get_default_config(self):
        return HDict(
            scheme                = None,
            model_name            = 'unnamed_model',
            distributed           = False,
            random_seed           = None,
            num_epochs            = 100,
            save_path             = HDict.L('c:path.join("models",c.model_name)'),
            checkpoint_path       = HDict.L('c:path.join(c.save_path,"checkpoint")'),
            config_path           = HDict.L('c:path.join(c.save_path,"config")'),
            summary_path          = HDict.L('c:path.join(c.save_path,"summary")'),
            log_path              = HDict.L('c:path.join(c.save_path,"logs")'),
            validation_frequency  = 1,
            batch_size            = HDict.L('c:128 if c.distributed else 32'),
            optimizer             = 'Adam'    ,
            max_lr                = 5e-4      ,
            clip_grad_value       = None      ,
            optimizer_params      = {}        ,
            dataloader_workers    = 0         ,
            dataloader_mp_context = 'forkserver',
            training_type         = 'normal'  ,
            evaluation_type       = 'validation',
            predictions_path      = HDict.L('c:path.join(c.save_path,"predictions")'),
            grad_accum_steps      = 1         ,
            prediction_bmult      = 1         ,
        )
    
    def get_default_state(self):
        state =  HDict(
            current_epoch = 0,
            global_step = 0,
        )
        return state
    
    def config_summary(self):
        if not self.is_main_rank: return
        for k,v in self.config.get_dict().items():
            print(f'{k} : {v}', flush=True)
    
    def save_config_file(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
        save_config_to_file(self.config.get_dict(), self.config.config_path+'.yaml')
        save_config_to_file(self.config_input, self.config.config_path+'_input.yaml')
    
    def model_summary(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.summary_path), exist_ok=True)
        trainable_params = 0
        non_trainable_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()
        summary = dict(
            trainable_params = trainable_params,
            non_trainable_params = non_trainable_params,
            model_representation = repr(self.model),
        )
        with open(self.config.summary_path+'.txt', 'w') as fp:
            yaml.dump(summary, fp, sort_keys=False, Dumper=yaml_Dumper)
    
    def save_checkpoint(self):
        if not self.is_main_rank: return
        ckpt_path = self.config.checkpoint_path
        os.makedirs(ckpt_path, exist_ok=True)
        
        torch.save(self.state, os.path.join(ckpt_path, 'training_state'))
        torch.save(self.base_model.state_dict(), os.path.join(ckpt_path, 'model_state'))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, 'optimizer_state'))
        print(f'Checkpoint saved to: {ckpt_path}',flush=True)
    
    def load_checkpoint(self):
        ckpt_path = self.config.checkpoint_path
        try:
            self.state.update(torch.load(os.path.join(ckpt_path, 'training_state')))
            self.base_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'model_state')))
            self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'optimizer_state')))
            if self.is_main_rank:
                print(f'Checkpoint loaded from: {ckpt_path}',flush=True)
            torch.cuda.empty_cache()
        except FileNotFoundError:
            pass
    
    # Callbacks
    def on_train_begin(self):
        pass
    def on_train_end(self):
        pass
    def on_epoch_begin(self, logs, training):
        pass
    def on_epoch_end(self, logs, training):
        pass
    def on_batch_begin(self, i, logs, training):
        pass
    def on_batch_end(self, i, logs, training):
        pass
    
    
    # Logging
    def get_verbose_logs(self):
        return OrderedDict(loss='0.4f')
    
    @cached_property
    def verbose_logs(self):
        return self.get_verbose_logs()
    
    def update_logs(self, logs, training, **updates):
        if training:
            logs.update(updates)
        else:
            logs.update(('val_'+k,v) for k,v in updates.items())
    
    def log_description(self, i, logs, training):
        if training:
            return list(f'{k} = {logs[k]:{f}}' 
                        for k,f in self.verbose_logs.items())
        else:
            return list(f'val_{k} = {logs["val_"+k]:{f}}' 
                        for k,f in self.verbose_logs.items())
    
    
    # Training loop
    def preprocess_batch(self, batch):
        if isinstance(batch, CollatedBatch):
            return CollatedBatch(self.preprocess_batch(b) for b in batch)
        elif hasattr(batch, 'cuda'):
            return batch.cuda(non_blocking=True)
        elif hasattr(batch, 'items'):
            return batch.__class__((k,v.cuda(non_blocking=True)) for k,v in batch.items())
        elif hasattr(batch, '__iter__'):
            return batch.__class__(v.cuda(non_blocking=True) for v in batch)
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')
    
    def calculate_loss(self, outputs, inputs):
        raise NotImplementedError
    
    def grad_accum_gather_outputs(self, outputs):
        return torch.cat(outputs, dim=0)
    
    def grad_accum_reduce_loss(self, loss):
        with torch.no_grad():
            total_loss = sum(loss)
        return total_loss
    
    def grad_accum_collator(self, dataloader):
        dataloader_iter = iter(dataloader)
        if self.config.grad_accum_steps == 1:
            yield from dataloader_iter
        else:
            while True:
                collated_batch = CollatedBatch()
                try:
                    for _ in range(self.config.grad_accum_steps):
                        collated_batch.append(next(dataloader_iter))
                except StopIteration:
                    break
                finally:
                    if len(collated_batch) > 0: yield collated_batch
    
    @cached_property
    def train_steps_per_epoch(self):
        if self.config.grad_accum_steps == 1:
            return len(self.train_dataloader)
        else:
            return (len(self.train_dataloader) + self.config.grad_accum_steps - 1)\
                            // self.config.grad_accum_steps
    
    @cached_property
    def validation_steps_per_epoch(self):
        return len(self.val_dataloader)
    
    
    def training_step(self, batch, logs):
        for param in self.model.parameters():
            param.grad = None
        
        if not isinstance(batch, CollatedBatch):
            outputs = self.model(batch)
            loss = self.calculate_loss(outputs=outputs, inputs=batch)
            loss.backward()
        else:
            num_nested_batches = len(batch)
            outputs = CollatedBatch()
            loss = CollatedBatch()
            
            sync_context = self.model.no_sync() \
                                if self.is_distributed else nullcontext()
            with sync_context:
                for b in batch:
                    o = self.model(b)
                    l = self.calculate_loss(outputs=o, inputs=b) / num_nested_batches
                    l.backward()
                    outputs.append(o)
                    loss.append(l)
            
            outputs = self.grad_accum_gather_outputs(outputs)
            loss = self.grad_accum_reduce_loss(loss)
        
        if self.config.clip_grad_value is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
        self.optimizer.step()
        return outputs, loss
    
    def validation_step(self, batch, logs):
        outputs = self.model(batch)
        loss = self.calculate_loss(outputs=outputs, inputs=batch)
        return outputs, loss
    
    def initialize_metrics(self, logs, training):
        pass
    
    def update_metrics(self, outputs, inputs, logs, training):
        pass
    
    def initialize_losses(self, logs, training):
        self._total_loss = 0.
    
    def update_losses(self, i, loss, inputs, logs, training):
        if not self.is_distributed:
            step_loss = loss.item()
        else:
            if training:
                loss = loss.detach()
            torch.distributed.all_reduce(loss)
            step_loss = loss.item()/self.ddp_world_size
        self._total_loss += step_loss
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/(i+1))
        
    
    def train_epoch(self, epoch, logs):
        self.model.train()
        self.initialize_losses(logs, True)
        self.initialize_metrics(logs, True)
        
        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)
        
        gen = self.grad_accum_collator(self.train_dataloader)
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True,
                       total=self.train_steps_per_epoch)
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch)
                outputs, loss = self.training_step(batch, logs)
                
                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)
                
                self.update_losses(i, loss, batch, logs, True)
                self.update_metrics(outputs, batch, logs, True)
                
                self.on_batch_end(i, logs, True)
                
                if self.is_main_rank:
                    desc = 'Training: '+'; '.join(self.log_description(i, logs, True))
                    gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
            for param in self.model.parameters():
                param.grad = None
    
    def minimal_train_epoch(self, epoch, logs):
        self.model.train()
        
        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)
            
        gen = self.grad_accum_collator(self.train_dataloader)
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True, desc='Training: ',
                       total=self.train_steps_per_epoch)
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch)
                _ = self.training_step(batch, logs)
                
                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)
                
                self.on_batch_end(i, logs, True)
        finally:
            if self.is_main_rank: gen.close()
            for param in self.model.parameters():
                param.grad = None
    
    
    def validation_epoch(self, epoch, logs):
        self.model.eval()
        self.initialize_losses(logs, False)
        self.initialize_metrics(logs, False)
        
        gen = self.val_dataloader
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True,
                       total=self.validation_steps_per_epoch)
        try:
            with torch.no_grad():
                for i, batch in enumerate(gen):
                    self.on_batch_begin(i, logs, False)
                    batch = self.preprocess_batch(batch)
                    outputs, loss = self.validation_step(batch, logs)
                    
                    self.update_losses(i, loss, batch, logs, False)
                    self.update_metrics(outputs, batch, logs, False)
                    
                    self.on_batch_end(i, logs, False)
                    
                    if self.is_main_rank:
                        desc = 'Validation: '+'; '.join(self.log_description(i, logs, False))
                        gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
    
    def load_history(self):
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        try:
            with open(history_file, 'r') as fp:
                return yaml.load(fp, Loader=yaml_Loader)
        except FileNotFoundError:
            return []
    
    def save_history(self, history):
        os.makedirs(self.config.log_path, exist_ok=True)
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        with open(history_file, 'w') as fp:
            yaml.dump(history, fp, sort_keys=False, Dumper=yaml_Dumper)

    
    def train_model(self):
        if self.is_main_rank: 
            history = self.load_history()
        starting_epoch = self.state.current_epoch
        
        self.on_train_begin()
        should_stop_training = False
        try:
            for i in range(starting_epoch, self.config.num_epochs):
                self.state.current_epoch = i
                if self.is_main_rank: 
                    print(f'\nEpoch {i+1}/{self.config.num_epochs}:', flush=True)
                logs = dict(epoch = self.state.current_epoch, 
                            global_step = self.state.global_step)
                
                try:
                    self.on_epoch_begin(logs, True)
                    if self.config.training_type == 'normal':
                        self.train_epoch(i, logs)
                    elif self.config.training_type == 'minimal':
                        self.minimal_train_epoch(i, logs)
                    else:
                        raise ValueError(f'Unknown training type: {self.config.training_type}')
                    self.on_epoch_end(logs, True)
                except StopTrainingException:
                    should_stop_training = True
                
                try:
                    if (self.val_dataloader is not None)\
                            and (not ((i+1) % self.config.validation_frequency)):
                        self.on_epoch_begin(logs, False)
                        if self.config.evaluation_type == 'validation':
                            self.validation_epoch(i, logs)
                        elif self.config.evaluation_type == 'prediction':
                            self.prediction_epoch(i, logs)
                        else:
                            raise ValueError(f'Unknown evaluation type: {self.config.evaluation_type}')
                    self.on_epoch_end(logs, False)
                except StopTrainingException:
                    should_stop_training = True
                
                self.state.current_epoch = i + 1
                if self.is_main_rank:
                    self.save_checkpoint()
                    
                    history.append(logs)
                    self.save_history(history)
                
                if should_stop_training:
                    if self.is_main_rank:
                        print('Stopping training ...')
                    break
        finally:
            self.on_train_end()
    
    def distributed_barrier(self):
        if self.is_distributed:
            dummy = torch.ones((),dtype=torch.int64).cuda()
            torch.distributed.all_reduce(dummy)
    
    # Prediction logic
    def prediction_step(self, batch):
        predictions = self.model(batch)
        if isinstance(batch, torch.Tensor):
            return dict(inputs=batch, predictions=predictions)
        elif isinstance(batch, list):
            outputs = batch.copy()
            batch.append(predictions)
            return outputs
        elif isinstance(batch, dict):
            outputs = batch.copy()
            outputs.update(predictions=predictions)
            return outputs
    
    def prediction_loop(self, dataloader):
        self.model.eval()
        
        outputs = []
        
        if self.is_main_rank:
            gen = tqdm(dataloader, dynamic_ncols=True)
        else:
            gen = dataloader
        try:
            with torch.no_grad():
                for batch in gen:
                    batch = self.preprocess_batch(batch)
                    outputs.append(self.prediction_step(batch))
        finally:
            if self.is_main_rank: gen.close()
        
        return outputs
    
    def preprocess_predictions(self, outputs):
        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], dict):
            return {k: torch.cat([o[k] for o in outputs], dim=0) 
                    for k in outputs[0].keys()}
        elif isinstance(outputs[0], list):
            return [torch.cat([o[i] for o in outputs], dim=0) 
                    for i in range(len(outputs[0]))]
        else:
            raise ValueError('Unsupported output type')
    
    def postprocess_predictions(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()
        elif isinstance(outputs, dict):
            return {k: v.cpu().numpy() for k, v in outputs.items()}
        elif isinstance(outputs, list):
            return [v.cpu().numpy() for v in outputs]
        else:
            raise ValueError('Unsupported output type')
    
    def distributed_gatther_tensor(self, tensors):
        shapes = torch.zeros(self.ddp_world_size+1, dtype=torch.long).cuda()
        shapes[self.ddp_rank+1] = tensors.shape[0]
        torch.distributed.all_reduce(shapes)
        
        offsets = torch.cumsum(shapes, dim=0)
        all_tensors = torch.zeros(offsets[-1], *tensors.shape[1:], dtype=tensors.dtype).cuda()
        all_tensors[offsets[self.ddp_rank]:offsets[self.ddp_rank+1]] = tensors
        
        torch.distributed.all_reduce(all_tensors)
        return all_tensors
    
    def distributed_gather_predictions(self, predictions):
        if self.is_main_rank:
            print('Gathering predictions from all ranks...')
        
        if isinstance(predictions, torch.Tensor):
            all_predictions = self.distributed_gatther_tensor(predictions)
        elif isinstance(predictions, list):
            all_predictions = [self.distributed_gatther_tensor(pred) for pred in predictions]
        elif isinstance(predictions, dict):
            all_predictions = {key:self.distributed_gatther_tensor(pred) 
                               for key, pred in predictions.items()}
        else:
            raise ValueError('Unsupported output type')
        
        if self.is_main_rank:
            print('Done.')
        return all_predictions
    
    def save_predictions(self, dataset_name, predictions):
        os.makedirs(self.config.predictions_path, exist_ok=True)
        predictions_file = os.path.join(self.config.predictions_path, f'{dataset_name}.pt')
        torch.save(predictions, predictions_file)
        print(f'Saved predictions to {predictions_file}')
    
    def evaluate_predictions(self, predictions):
        raise NotImplementedError
    
    def prediction_epoch(self, epoch, logs):
        if self.is_main_rank:
            print(f'Predicting on validation dataset...')
        dataloader = self.val_dataloader
        outputs = self.prediction_loop(dataloader)
        outputs = self.preprocess_predictions(outputs)
    
        if self.is_distributed:
            outputs = self.distributed_gather_predictions(outputs)
        
        predictions = self.postprocess_predictions(outputs)
        if self.is_main_rank:
            self.save_predictions('validation', predictions)
        results = self.evaluate_predictions(predictions)
        results = {f'val_{k}': v for k, v in results.items()}
        logs.update(results)
        if self.is_main_rank:
            desc = 'Validation: '+'; '.join(f'{k}: {v:.4f}' for k, v in results.items())
            print(desc, flush=True)
            
    
    # Interface
    def prepare_for_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_checkpoint()
        self.model_summary()
        
    def execute_training(self):
        self.prepare_for_training()
        self.train_model()
        self.finalize_training()
    
    def finalize_training(self):
        pass
    
        
