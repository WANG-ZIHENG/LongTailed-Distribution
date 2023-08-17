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

  