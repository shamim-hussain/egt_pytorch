# Edge-augmented Graph Transformer (PyTorch)

## News

* 06/21/2022 - The trained checkpoints on the [PCQM4M-V2](https://ogb.stanford.edu/docs/lsc/) have been released. They are available at <https://zenodo.org/record/6680242>. For additional information, see the ["Download Trained Model Checkpoints"](#download-trained-model-checkpoints) section below.
* 06/05/2022 - The [accepted prerprint](https://arxiv.org/abs/2108.03348) our paper in KDD '22 is now available on arXiv. It includes discussions on dynamic centrality scalers, random masking, attention dropout and other details about the latest experiments and results. Note that the title is changed to **"Global Self-Attention as a Replacement for Graph Convolution"**.
* 05/18/2022 - Our paper "Global Self-Attention as a Replacement for Graph Convolution" has been accepted at [KDD'22](https://kdd.org/kdd2022/). The preprint at arXiv will be updated soon with the latest version of the paper.

## Introduction

This is the official **PyTorch** implementation of the **Edge-augmented Graph Transformer (EGT)** as described in <https://arxiv.org/abs/2108.03348>, which augments the Transformer architecture with residual edge channels. The resultant architecture can directly process graph-structured data. For a **Tensorflow** implementation see: <https://github.com/shamim-hussain/egt>.

This implementation focuses on the [OGB-Mol](https://ogb.stanford.edu/docs/graphprop/) datasets and [OGB-LSC](https://ogb.stanford.edu/docs/lsc/) datasets. (OGB-Mol datasets utilize transfer learning from PCQM4Mv2 dataset.)

## Results

Dataset       | #layers | #params | Metric         | Valid           | Test           |
--------------|---------|---------|----------------|-----------------|----------------|
PCQM4M        | 18      | 47.4M   | MAE            | 0.1225          | --             |
PCQM4M-V2     | 18      | 47.4M   | MAE            | 0.0883          | --             |
PCQM4M-V2     | 24      | 89.3M   | MAE            | 0.0857          | 0.0862         |
OGBG-MolPCBA  | 30      | 110.8M  | Avg. Precision | 0.3021 ± 0.0053 | 0.2961 ± 0.0024|
OGBG-MolHIV   | 30      | 110.8M  | ROC-AUC        | 0.8060 ± 0.0065 | 0.8051 ± 0.0030|

## Download Trained Model Checkpoints

The trained model checkpoints on the PCQM4M-V2 dataset are available at <https://zenodo.org/record/6680242>. Individual *zip* files are downloadable. The extracted folders can be put under the *models/pcqm4mv2* directory. See the *config_input.yaml* file contained within to see the training configurations.

We found that the results can be further improved by freezing the node channel layers and training the edge channel layers for a few additional epochs. The corresponding tuned models are given the suffix **-T** and achieve better results than their untuned counterparts. However, its effect on transfer learning has not yet been studied. That is why we include checkpoints for both tuned and untuned models.

Model            | #layers | #params | Valid MAE       | Test MAE       | Comment                                |
-----------------|---------|---------|-----------------|----------------|----------------------------------------|
EGT-48M-SIMPLE   | 18      | 47.2M   | 0.0872          | --             | EGT-Simple (lightweight variant of EGT)|
EGT-48M-SIMPLE-T | 18      | 47.2M   | 0.0860          | --             | Tuned version of above                 |
EGT-90M          | 24      | 89.3M   | 0.0869          | 0.0872         | **Submitted to the leaderboard**       |
EGT-90M-T        | 24      | 89.3M   | **0.0857**      | **0.0862**     | **Submitted tuned version of above**   |
EGT-110M         | 30      | 110.8M  | 0.0870          | --             | **Used for transfer learning**         |
EGT-110M-T       | 30      | 110.8M  | 0.0859          | --             | Tuned version of above                 |

## Requirements

* `python >= 3.7`
* `pytorch >= 1.6.0`
* `numpy >= 1.18.4`
* `numba >= 0.50.1`
* `ogb >= 1.3.2`
* `rdkit>=2019.03.1`
* `yaml >= 5.3.1`

## Run Training and Evaluations

You can specify the training/prediction/evaluation configurations by creating a `yaml` config file and also by passing a series of `yaml` readable arguments. (Any additional config passed as argument willl override the config specified in the file.)

* To run training: ```python run_training.py [config_file.yaml] ['config1: value1'] ['config2: value2'] ...```
* To make predictions: ```python make_predictions.py [config_file.yaml] ['config1: value1'] ['config2: value2'] ...```
* To perform evaluations: ```python do_evaluations.py [config_file.yaml] ['config1: value1'] ['config2: value2'] ...```

Config files for the results can be found in the configs directory. Examples:
```
python run_training.py configs/pcqm4m/egt_47m.yaml
python run_training.py 'scheme: pcqm4m' 'model_height: 6'
python make_predictions.py configs/pcqm4m/egt_47m.yaml 'evaluate_on: ["val"]'
```

### More About Training

Once the training is started a model folder will be created in the *models* directory, under the specified dataset name. This folder will contain a copy of the input config file, for the convenience of resuming training/evaluation. Also, it will contain a config.yaml which will contain all configs, including unspecified default values, used for the training. Training will be checkpointed per epoch. In the case of any interruption, you can resume training by running the *run_training.py* with the config.yaml file again.

### Configs
There many different configurations. The only **required** configuration is `scheme`, which specifies the training scheme. If the other configurations are not specified, a default value will be assumed for them. Here are some of the commonly used configurations:

`scheme`: pcqm4m/pcqm4mv2/molpcba/mohiv.

`dataset_path`: Where the downloaded OGB datasets will be saved.

`model_name`: Serves as an identifier for the model, also specifies default path of the model directory, weight files etc.

`save_path`: The training process will create a model directory containing the logs, checkpoints, configs, model summary and predictions/evaluations. By default it creates a folder at *models/<dataset_name>* but it can be changed via this config.

`cache_dir`: During first time of training/evaluation the data will be cached. Default path is *cache_data/<dataset_name>*. But it can be changed via this config.

`distributed`: In a multi-gpu setting you can set it to True, for distributed training. Note that, the batch size should also be adjusted accordingly.

`batch_size`: Batch size. In case of distributed training it is the local batch size. So, the total batch size = batch_size x number of available gpus.

`num_epochs`: Maximum Number of epochs.

`max_lr`: Maximum learning rate.

`min_lr`: Minimum learning rate.

`lr_warmup_steps`: Initial linear learning rate warmup steps.

`lr_total_steps`: Total number of gradient updates to be performed, including linear warmup and cosine decay.

`model_height`: The number of layers *L*.

`node_width`: The dimensionality of the node channels *d_h*.

`edge_width`: The dimensionality of the edge channels *d_e*.

`num_heads`: The number of attention heads. Default is 8.

`node_ffn_multiplier`: FFN multiplier for node channels.

`edge_ffn_multiplier`: FFN multiplier for edge channels.

`virtual_nodes`: number of virtual nodes. 0 (default) would result in global average pooling being used instead of virtual nodes.

`upto_hop`: Clipping value of the input distance matrix.

`attn_dropout`: Dropout rate for the attention matrix.

`node_dropout`: Dropout rate for the node channel's MHA and FFN blocks.

`edge_dropout`: Dropout rate for the edge channel's MHA and FFN blocks.

`sel_svd_features`: Rank of the SVD encodings *r*.

`svd_calculated_dim` : Number of left and right singular vectors calculated and cached for svd encodings.

`svd_output_dim` : Number of left and right singular vectors used as svd encodings.

`svd_random_neg` : Whether to randomly flip the signs of the singular vectors. Default - true.

`pretrained_weights_file` : Used to specify the learned weights of an already trained model.

## Python Environment

The Anaconda environment in which the experiments were conducted is specified in the `environment.yml` file.

## Citation

Please cite the following paper if you find the code useful:
```
@article{hussain2021global,
  title={Global Self-Attention as a Replacement for Graph Convolution},
  author={Hussain, Md Shamim and Zaki, Mohammed J and Subramanian, Dharmashankar},
  journal={arXiv preprint arXiv:2108.03348},
  year={2021}
}
```
