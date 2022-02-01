

## Dataset

GTEA ：[paper](http://amav.gatech.edu/sites/default/files/papers/cvpr2011.Fathi_.Ren_.Rehg_.printed.pdf)

50Salads:  [paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.8684&rep=rep1&type=pdf)

Breakfast:  [paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kuehne_The_Language_of_2014_CVPR_paper.pdf)

You can download features and G.T. of these datasets from [this repository](http://pan.dlut.edu.cn/share?id=xgja6ssj99rt) or [this repository](https://github.com/yabufarha/ms-tcn).

## Requirements

~~~python
* Python 3.x
* pytorch => 1.0
* torchvision
* pandas
* numpy
* tqdm
* PyYAML
* addict
~~~

**You can download packages using requirements.txt.**

~~~pip3
pip3 install -r requirements.txt
~~~

## directory structure

~~~python 
basestation_rust_detection ── csv/
                           ├─ libs/
                           ├─ result/
                           ├─ utils/
                           ├─ dataset ── 50salads/...
                           │           ├─ breakfast/...
                           │           └─ gtea ── features/
                           │                    ├─ groundTruth/
                           │                    ├─ splits/
                       	   │                    ├─ gt_arr/
                           │                    └─ mapping.txt
                           ├.gitignore
                           ├ README.md
                           ├ requirements.txt
                           ├ train.py
                           ├ train.sh
                           ├ eval.py
                           └ eval.sh
~~~



## How to use

### Training

Run：

~~~python3
python3 train.py --config ./result/XXX/ms-tcn/XXX/config.yaml
or:
bash train.sh
~~~

You can train a model in your own setting. Follow the below example of a configuration file.

~~~yaml
model: ms-tcn
stages: ['dilated', 'dilated', 'dilated', 'dilated']
n_features: 64
dilated_n_layers: 10
kernel_size: 15

# loss function
ce: True    # cross entropy
tmse: True    # temporal mse
tmse_weight: 0.15

class_weight: True    # if you use class weight to calculate cross entropy or not

batch_size: 1

# the number of input feature channels
in_channel: 2048

# thresholds for calcualting F1 Score
thresholds: [0.1, 0.25, 0.5]

num_workers: 4
max_epoch: 50

optimizer: Adam
scheduler: None

learning_rate: 0.0005
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001  # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound
poly_power: 0.9       # for polunomial learning scheduler

param_search: True		# if you do validation to determine hyperparams
param_groups: True		# if you do validation to determine hyperparams
device: 1				# GPU 

dataset: 50salads
dataset_dir: ./dataset
csv_dir: ./csv
split: 3

result_path: ./result/50salads/ms-tcn/split3
~~~

### Test

~~~python
python3 eval.py ./result/XXX/ms-tcn/XXX/config.yaml test
or:
bash eval.sh
~~~

### Average cross validation results

~~~python
python3 utils/average_cv_results.py ./result/XXX/ms-tcn/
~~~


## References

* code_based_on [leslieAIbin/ms-tcn-pytorch](https://github.com/leslieAIbin/ms-tcn-pytorch)
