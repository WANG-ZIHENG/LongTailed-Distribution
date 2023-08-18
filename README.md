# Optimizing Accuracy for Image Classification under Long-tail Distribution

### Requirements

The "requirements.txt" file required to configure the environment is in the folder of the corresponding method.

```
pip install -r requirements.txt
```



### Datasets

[mini-ImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-ImageNet)

Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning[J]. Advances in neural information processing systems, 2016, 29.

```
data
└──mini-ImageNet
    ├── train
    ├── val
    └── test
```



[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images[J]. 2009.

```
data
└──CIFAR10
    ├── train
    ├── val
    └── test
```

```
data
└──CIFAR100
    ├── train
    ├── val
    └── test
```

### Loss Functions Experiment Reproduction

```python
# Train

# Open the 'imagenet_lt_train.py' file
# All methods and datasets have been made configurable through the addition of arguments to the parser.

# change training epochs
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')

# change the default path as 'data/CIFAR10' or 'data/CIFAR100'
parser.add_argument('--data_dir', type=str, default='data/mini-ImageNet')

# when test CIFAR10, it only has 10 classes
parser.add_argument('--cls_num', type=int, default=100)

# when default=0, the class weights of all classes are 1
# this modulates the Class-Balanced Loss beta parameter
# it was set default=0.999 in this experiment
parser.add_argument('--cbl_beta', default=0.999, type=float,
                    help='Beta for class balanced loss.')

# when default=True, the last fully connected layer is "NormedLinear"
# when default=False, the last fully connected layer is "Linear"
parser.add_argument('--norm_out', type=bool, default=True)

# when default=False, the Rare Sample Generator will be removed
parser.add_argument('--add_rsg', type=bool, default=True)

# taking LDAM as an example
# LDAM has a parameter named "weight"
# when default='icf', it means weight of LDAM calculated by inverse class frequency, α-weighted
# when default='cbl', it means weight of LDAM calculated by effective numbers of class, CB-weighted
# when default='cbl' and '--cbl_beta, default=0', unweighted
parser.add_argument('--ldam_weight_type', type=str, choices=['cbl', 'icf'], default='icf')

# Loss = 0.1 * LDAM + 0.9 * SupCon
parser.add_argument('--ldam_loss_weight', type=float, default=0.1)
parser.add_argument('--supcon_loss_weight', type=float, default=0.9)

# all loss functions below, if default=True, corresponding loss functions will be activated.
parser.add_argument('--add_cross_entropy_loss', type=bool, default=False)
parser.add_argument('--add_ldam_loss', type=bool, default=True)
parser.add_argument('--add_focal_loss', type=bool, default=False)
parser.add_argument('--add_supcon_loss', type=bool, default=True)
parser.add_argument('--add_arc_margin_loss', type=bool, default=False)
parser.add_argument('--add_add_margin_loss', type=bool, default=False)
parser.add_argument('--add_sphere_loss', type=bool, default=False)
```

```python
# Test

# Open the 'imagenet_lt_test.py' file
# change dataset name used
parser.add_argument('--data_dir', type=str, default='data/mini-ImageNet')

# the trained model weights are loaded from the specified "model_dir"
parser.add_argument('--model_dir', type=str,
 default='ImageNet_LT_CIFAR10_resnext50_32x4d_normout_rsg_ldam_supcon')
```



### Referenced Technology

*  Zhang Y, Kang B, Hooi B, et al. Deep long-tailed learning: A survey[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

   [Rare Sample Generator](https://github.com/Vanint/Awesome-LongTailed-Learning)

* Rombach R, Blattmann A, Lorenz D, et al. High-resolution image synthesis with latent diffusion models[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 10684-10695.

​		[Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)

* Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.

  [Focal Loss](https://github.com/facebookresearch/detectron)

* Cui Y, Jia M, Lin T Y, et al. Class-balanced loss based on effective number of samples[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9268-9277.

  [Class-Balanced Loss](https://github.com/richardaecn/class-balanced-loss)

* Cao K, Wei C, Gaidon A, et al. Learning imbalanced datasets with label-distribution-aware margin loss[J]. Advances in neural information processing systems, 2019, 32.

  [Label-Distribution-Aware Margin Loss](https://github.com/kaidic/LDAM-DRW)

  

* Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning[J]. Advances in neural information processing systems, 2020, 33: 18661-18673.

  [Supervised Contrastive Loss](https://github.com/HobbitLong/SupContrast)

* Liu W, Wen Y, Yu Z, et al. Sphereface: Deep hypersphere embedding for face recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 212-220.

  [SphereFace](https://github.com/wy1iu/sphereface)

* Deng J, Guo J, Xue N, et al. Arcface: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4690-4699.

  [Additive Angular Margin Loss](https://github.com/deepinsight/insightface)

* Wang F, Cheng J, Liu W, et al. Additive margin softmax for face verification[J]. IEEE Signal Processing Letters, 2018, 25(7): 926-930.

  [Additive Margin Softmax](https://github.com/happynear/AMSoftmax)

  