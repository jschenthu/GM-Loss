This is the codebase of the GM loss on MNIST and CIFAR10 datasets.

# Environment settings
The codebase is tested under the following environment settings:
- cuda: 11.0
- python: 3.8.10
- pytorch: 1.7.1
- torchvision: 0.8.2

# MNIST
The code uses the network of 6 convolutional layers and a linear layer with 2-D / 100-D output.

## 2-D

To train the 2-D model with GM loss, please run the following code inside the ./mnist/ folder. The training procedure needs around 20 minutes on a Titan X gpu.

```shell
CUDA_VISIBLE_DEVICES=0 python mnist_release.py --loss gm --lr 0.01 --edim 2
```

In the default setting, $\alpha$ is set to be 1.0 and $\lambda$ is set to be 0.1.

The best accuracy is around 99.29% for GM loss.

After running the training code, the feature distribution on the testing set will be shown in ./mnist/mnist_gm_test.png.

If you want to train with the softmax loss, please run the following code inside the ./mnist/ folder. You will get the best accuracy of around 99.02%. 

```shell
CUDA_VISIBLE_DEVICES=0 python mnist_release.py --loss softmax --lr 0.01 --edim 2
```


## 100-D

To train the 100-D model with GM loss, please run the following code inside the ./mnist/ folder. The training procedure needs around 20 minutes on a Titan X gpu.

```shell
CUDA_VISIBLE_DEVICES=0 python mnist_release.py --loss gm --lr 0.02 --edim 100
```

In the default setting, $\alpha$ is set to be 1.0 and $\lambda$ is set to be 0.1.

The best accuracy is around 99.70% for GM loss.

See Table 1 in the paper for reference.

# CIFAR10
The code uses the ResNet20 network. 

To train the model with GM loss, please run the following code inside the ./cifar/ folder. The training procedure needs around 70 minutes on a Titan X gpu.

```shell
CUDA_VISIBLE_DEVICES=0 python cifar10_release.py --loss gm
```
In the default setting, $\alpha$ is set to be 0.3 and $\lambda$ is set to be 0.1.

The best accuracy is around 92.80% for GM loss.

See Table 2 in the paper for reference.