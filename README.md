# README

Overview
--------
This workspace contains three related codebases:

- [TagFex_CVPR2025](TagFex_CVPR2025/) — original TagFex (CVPR 2025) source code.
- [l2p-tagfex](l2p-tagfex/) — my experiment integrating the L2P module into TagFex.
- [l2p-pytorch](l2p-pytorch/) — orginal L2P PyTorch implementation used for the prompt module and ViT utilities.

Project structure
-----------------
Top-level files and folders:

- [README.md](README.md) (this file)  
- [l2p-pytorch/](l2p-pytorch/)
  - [l2p-pytorch/README.md](l2p-pytorch/README.md)
  - [l2p-pytorch/main.py](l2p-pytorch/main.py)
  - [l2p-pytorch/vision_transformer.py](l2p-pytorch/vision_transformer.py)
  - [l2p-pytorch/run_with_submitit.py](l2p-pytorch/run_with_submitit.py)
  - [l2p-pytorch/prompt.py](l2p-pytorch/prompt.py)
  - [l2p-pytorch/utils.py](l2p-pytorch/utils.py)
  - [l2p-pytorch/requirements.txt](l2p-pytorch/requirements.txt)
  - training scripts: [l2p-pytorch/train_cifar100_l2p.sh](l2p-pytorch/train_cifar100_l2p.sh), [l2p-pytorch/train_five_datasets_l2p.sh](l2p-pytorch/train_five_datasets_l2p.sh)
- [l2p-tagfex/](l2p-tagfex/)
  - [l2p-tagfex/main.py](l2p-tagfex/main.py)
  - [l2p-tagfex/main-tagfex.py](l2p-tagfex/main-tagfex.py)
  - [l2p-tagfex/tagfex.py](l2p-tagfex/tagfex.py)
  - [l2p-tagfex/prompt.py](l2p-tagfex/prompt.py)
  - [l2p-tagfex/model.py](l2p-tagfex/model.py)
  - [l2p-tagfex/resnet18.py](l2p-tagfex/resnet18.py)
  - [l2p-tagfex/tagfexnet.py](l2p-tagfex/tagfexnet.py)
- [TagFex_CVPR2025/](TagFex_CVPR2025/)
  - [TagFex_CVPR2025/README.md](TagFex_CVPR2025/README.md)
  - original TagFex implementation and configs (see folder)
- Other helper code and datasets:
  - [configs/](configs/) (shared or per-method configs)
  - [continual_datasets/](continual_datasets/) and [data/](l2p-tagfex/data/)
  - [output/](l2p-pytorch/output/), [logs/](l2p-tagfex/logs/), [loggers/](l2p-tagfex/loggers/)
  - method implementations: [l2p-tagfex/methods/](l2p-tagfex/methods/) and [TagFex_CVPR2025/methods/](TagFex_CVPR2025/methods/)

Notes
-----
- The original TagFex code is preserved in [TagFex_CVPR2025/](TagFex_CVPR2025/). Use its README and configs for reproducing the original paper's baseline experiments: [TagFex_CVPR2025/README.md](TagFex_CVPR2025/README.md).
- My integrated experiments live in [l2p-tagfex/](l2p-tagfex/). This includes the integration of the L2P prompt module and modified training scripts.
- L2P utilities (ViT, prompt implementation, and submitit support) are in [l2p-pytorch/](l2p-pytorch/). See [l2p-pytorch/vision_transformer.py](l2p-pytorch/vision_transformer.py) and [l2p-pytorch/prompt.py](l2p-pytorch/prompt.py).

Requirements
------------
- TagFex_CVPR2025 (original): pytorch, torchvision, torchmetrics, loguru, tqdm (see [TagFex_CVPR2025/README.md](TagFex_CVPR2025/README.md)).
- l2p-pytorch: see [l2p-pytorch/requirements.txt](l2p-pytorch/requirements.txt).
- l2p-tagfex: requires dependencies from both TagFex and l2p-pytorch (PyTorch, timm, etc.). Inspect [l2p-pytorch/requirements.txt](l2p-pytorch/requirements.txt) and TagFex reqs.

Quick start — prepare environment
---------------------------------
1. Create and activate a Python environment (recommended Python 3.8+).
2. Install TagFex dependencies:
   - pip install torch torchvision torchmetrics loguru tqdm
3. Install L2P dependencies:
   - pip install -r l2p-pytorch/requirements.txt

Running experiments
-------------------
1. Original TagFex:
   - cd TagFex_CVPR2025
   - Example:
     - python main.py train --exp-configs configs/all_in_one/cifar100_10-10_tagfex_resnet18.yaml --log-dir ./logs/exp_cifar100_10-10

2. My integrated experiments (l2p-tagfex):
   - cd l2p-tagfex
   - Ensure dependencies from TagFex_CVPR2025 and l2p-pytorch are installed.
   - Example single-node run:
     - python main-tagfex.py train --exp-configs configs/all_in_one/cifar100_10-10_tagfex_resnet18.yaml --log-dir ./logs/l2p_tagfex_exp --pool_size 7

Reproducing results
-------------------
1. Use the same configs used in experiments (located under each repo's `configs/` directory).
2. Fix random seeds and environment settings in config files if replication requires deterministic runs.
3. For large-scale experiments, run multi-node training as described in [TagFex_CVPR2025/README.md](TagFex_CVPR2025/README.md) or [l2p-pytorch/run_with_submitit.py](l2p-pytorch/run_with_submitit.py).

Pointers to key code
--------------------
- L2P ViT/init utilities: [l2p-pytorch/vision_transformer.py](l2p-pytorch/vision_transformer.py)  
- Integrated TagFex + L2P entry point: [l2p-tagfex/main-tagfex.py](l2p-tagfex/main-tagfex.py)  
- TagFex method implementation: [TagFex_CVPR2025/methods/tagfex/tagfexnet.py](TagFex_CVPR2025/methods/tagfex/tagfexnet.py) and my variant [l2p-tagfex/tagfexnet.py](l2p-tagfex/tagfexnet.py)

Contributing / Notes
--------------------
- The integration in [l2p-tagfex/](l2p-tagfex/) reuses L2P prompt modules from [l2p-pytorch/](l2p-pytorch/) and adapts TagFex model/training code. Review the prompt and ViT code in [l2p-pytorch/vision_transformer.py](l2p-pytorch/vision_transformer.py) and [l2p-pytorch/prompt.py](l2p-pytorch/prompt.py).
- I used Resnet18 as a backbone instead of ViT. Corresponding Resnet18 backbone can be found in [l2p-tagfex/resnet18.py](l2p-tagfex\resnet18.py)
- The log files of the replication can be found here: ["TagFex_CVPR2025\logs\exp_cifar100_10-10-2\exp_stdlog0.log"](TagFex_CVPR2025\logs\exp_cifar100_10-10-2\exp_stdlog0.log)
- The log files of the L2P module in Tagfex can be found here: [l2p-tagfex\logs\exp_cifar100_10-10-test\exp_stdlog0.log](l2p-tagfex\logs\exp_cifar100_10-10-test\exp_stdlog0.log)
- The plot of parameter count vs number of tasks can be found here: [plots\parameters_per_task.png](plots\parameters_per_task.png)
