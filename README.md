### Rectified Lagrangian for Out-of-Distribution Detection in Modern Hopfield Networks

The document is guidance for reproducing [our paper](https://www.arxiv.org/abs/2502.14003), some codes are from [energy-ood](https://github.com/wetliu/energy_ood) and [SHE_ood_detection](https://github.com/zjs975584714/SHE_ood_detection).

#### Abstract
Modern Hopfield networks (MHNs) have recently gained significant attention in the field of artificial intelligence because they can store and retrieve a large set of patterns with an exponentially large memory capacity.
A MHN is generally a dynamical system defined with Lagrangians of memory and feature neurons,
where memories associated with in-distribution (ID) samples are represented by attractors in the feature space.
One major problem in existing MHNs lies in managing out-of-distribution (OOD) samples because it was originally assumed that all samples are ID samples.
To address this, we propose the rectified Lagrangian
(RegLag), a new Lagrangian for memory neurons
that explicitly incorporates an attractor for OOD samples in the dynamical system of MHNs.
RecLag creates a trivial point attractor for any interaction matrix, enabling OOD detection by identifying samples that fall into this attractor as OOD. 
The interaction matrix is optimized so that the probability densities can be estimated to identify ID/OOD.
We demonstrate the effectiveness of RecLag-based MHNs compared to energy-based OOD detection methods, including those using state-of-the-art Hopfield energies, across nine image datasets.

#### Poster
[aaai2025_moriai_poster.pdf](https://github.com/user-attachments/files/18947439/aaai2025_moriai_poster.pdf)


#### Preliminaries
Our code is tested under Ubuntu Linux 18.04.1 and Python 3.6 environment. The environment can be accomplished by the following command:
```
pip install -r requirement.txt
```

#### Download the Out-of-distribution (OOD) Dataset
In our paper, we use nine OOD datasets and two In-distribution (ID) datasets.
 For the ID datasets CIFAR10 and CIFAR100 and one of the OOD datasets SVHN, it is easy to use them directly by the ```torchvision``` as follows (```data_path``` refers to your specified dataset path):
```
dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
dataset = torchvision.datasets.CIFAR100(root=data_path, train=False=True)
dataset = torchvision.datasets.SVHN(root=data_path, train=False, download=True)
```
However, another eight OOD datasets need to download and we provide the link to them.
[LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz), [Places](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/),[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/),[Tiny Imagenet](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz),[SUN](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/),[iNaturalist](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/)
Please place them into your dataset path, and use them as follows:
```
dataset =  torchvision.datasets.ImageFolder(root=data_path)
```
### Pretrained Model
We use ResNet18, ResNet34, and WRN40-2 as our backbone networks. And the pre-trained models are all trained on CIFAR10 and CIFAR100 respectively.  As mentioned in our paper, to get better performance, we use data augmentation (e.g., flip, rotate) and resize the image during the training/testing process. We use size = 112 and 64 for ResNet and WRN respectively.

### Evaluation process
##### (1) Prepare the model
All the model weights have been saved at ```./checkpoints/cifar10/``` and ```./checkpoints/cifar100/```, you can use them directly if you do not want to train them again.
Otherwise, you can train the model by the following command:
```
python train_model.py --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --dataset_path xxx(data_path)
```
##### (2) Prepare the stored pattern we need
Stored pattern must be prepared in advance to run the our or comparative method. There are two types, “SHE” and “HE”, especially “HE” needs to be saved for the execution of our method
```
python generate_stored_pattern.py  --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --score (SHE/HE) --stored_data_path xxx(data_path)
```
After several time, the stored pattern is stored at ```./stored_pattern/all_stored_pattern``` and ```./stored_pattern/avg_stored/pattern/```.

##### (3) Run our method
You can evaluate our method by run the following command:
```
python learn_RecLag.py --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --stored_data_path xxx(data_path)
python test_RecLag.py --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --stored_data_path xxx(data_path)
```

In addition, the comparison method runs as follows
```
python test_score_ood_detection.py --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --score(MSP/Energy/ReAct/HE/SHE) --stored_data_path xxx(data_path)
```
