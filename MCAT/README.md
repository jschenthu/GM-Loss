# Joint Classification and Adversarial Detection

This repository contains the PyTorch code and a fast implementation for our proposed **Mahalanobis distance Calibrated Adversarial Training (MCAT)**,
which performs joint classification and detection against white-box adversarial attacks.

## Installation

This repository requires, among others, the following packages:

* Python >=3.5
* [PyTorch](https://pytorch.org/) >= 1.1 and [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
* [Tensorflow](https://www.tensorflow.org/) for [Tensorboard](https://www.tensorflow.org/tensorboard)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [h5py](https://www.h5py.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/)
* [imageio](https://imageio.github.io/)
* [imgaug](https://scikit-image.org/docs/dev/api/skimage.html)
* [iPython](https://ipython.org/) and [Jupyter](https://jupyter.org/install) (for evaluation)
* [wget](https://pypi.org/project/wget/) (for examples)

Running

    python setup.py

can be used to check whether all requirements are met.
The script also checks paths to data and experiments required for
reproducing the experiments.

## Dataset

Change directory to `experiments`, and run

    python download_dataset.py cifar10

The CIFAR10 dataset will be downloaded in `work_space/data`.

## Training

To train MCAT with on `GPU 0`, change directory to `experiments`, and run

    CUDA_VISIBLE_DEVICES=0 python train.py MahalanobisdistanceCalibratedAdversarialTrainingInterface config.cifar10 mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7

Training can be monitored using

    python train_board.py config.cifar10 mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7 -port <port>

which will start a TensorBoard session on the provided port.

The checkpoints of the model will be saved in `work_space/experiments/Cifar10/mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7`.

## Evaluation

To attack the model trained by MCAT on `GPU 0`, change directory to `experiments`, and run

    CUDA_VISIBLE_DEVICES=0 python attack.py config.cifar10 mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7 set_test

The adversarial examples will be saved in `work_space/experiments/Cifar10/mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7`.

For evaluation on `GPU 0`, run

    CUDA_VISIBLE_DEVICES=0 python test.py config.cifar10 mcat_power2_10_alpha0.1_lambda0.050_beta0.5_thres0.7 set_test

The output of the command is approximately as follows

    Error clean (%) : 8.3
    Threshold distance : 1.31
    Minimum d_xy : 2.78
    
    Robust Errors (%) for 99% TPR
    L_inf (epsilon=0.3) : 60.1
    L_inf (epsilon=0.4) : 77.5
    L_2 (epsilon=3) : 67.5
    L_1 (epsilon=18) : 56.3
    L_0 (epsilon=15) : 30.8
    Adv Frames : 63.4
    Average : 59.3
    Worst Case : 77.5


