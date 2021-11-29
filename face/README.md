# Face Verification Experiments of GM Loss in Pytorch
## Requirements
1. Install pytorch(torch>=1.6.0).
2. `pip install -r requirements.txt`.
3. Download `faces_vgg_112.zip` from [https://cloud.tsinghua.edu.cn/f/d02a10ba5e6243ab9e85/?dl=1](https://cloud.tsinghua.edu.cn/f/d02a10ba5e6243ab9e85/?dl=1), extract it, and put them into `./train_tmp/`.
4. Download `ijb-testsuite.tar` from [https://cloud.tsinghua.edu.cn/f/12caebb5a3eb4b259fc4/?dl=1](https://cloud.tsinghua.edu.cn/f/12caebb5a3eb4b259fc4/?dl=1), extract it, and put it into `./`.
## Train and Test
In our experiments, ResNet50 model is trained on VGGFace2 and verified on IJB-B and IJB-C. Here are two examples of running our codes:

To train and test using our GM loss on `GPU 0,1,2,3`, please run `bash run_gm.sh`.

## Expected Results
ResNet50 with GM loss on IJB-B:
<table>
    <tr>
        <th>FPR</th>
        <td>1e-6</td>
        <td>1e-5</td>
        <td>1e-4</td>
        <td>0.001</td>
        <td>0.01</td>
        <td>0.1</td>
    </tr>
    <tr>
        <th>TPR(%)</th>
        <td>48.90</td>
        <td>82.08</td>
        <td>90.93</td>
        <td>95.25</td>
        <td>97.80</td>
        <td>99.06</td>
    </tr>
</table>
ResNet50 with GM loss on IJB-C:
<table>
    <tr>
        <th>FPR</th>
        <td>1e-6</td>
        <td>1e-5</td>
        <td>1e-4</td>
        <td>0.001</td>
        <td>0.01</td>
        <td>0.1</td>
    </tr>
    <tr>
        <th>TPR(%)</th>
        <td>79.79</td>
        <td>87.46</td>
        <td>92.98</td>
        <td>96.35</td>
        <td>98.40</td>
        <td>99.33</td>
    </tr>
</table>





