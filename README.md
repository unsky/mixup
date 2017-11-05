# mixup

This is an implement and Improvement  on mixup: Beyond Empirical Risk Minimization https://arxiv.org/abs/1710.09412

# The improvement 

1. add backward
2. add mix rate


### Two scenes:
![image](https://github.com/unsky/mixup/blob/master/3.png)


### The detail design of mixUPlayer:

![image](https://github.com/unsky/mixup/blob/master/4.png)


# The results:


|         cifar10               | alpha         | mix_rate  | test Acc |initial learning rate|
| -------------          |:-------------:| -----:      | -----:   | -----:  |
| (ERM)resnet50 90epoch  |      -        |-            | 0.87900390625  | 0.05|
|(ERM)resnet50 200epoch  |      -        |-            | 0.89365234375 | 0.05|
|(ERM)resnet50 300epoch  |      -        |-            | 0.8931640625|0.05 |
| (mixup)resnet50 90epoch|      0.2     |0.7           |0.8609375      | 0.7|
| (mixup)resnet50 200epoch|      0.2     |0.7           |0.91611328125      | 0.7|
| (mixup)resnet50 300epoch|      0.2     |0.7          | 0.9224609375     | 0.7|
| mixup in feature maps（resnet50 head conv）90epoch|      0.2     |0.7          | 0.8544921875  |0.7 |
| mixup in feature maps（resnet50 head conv）200epoch|      0.2     |0.7          | 0.91796875  |0.7 |
| mixup in feature maps（resnet50 head conv）300epoch|      0.2     |0.7          | 0.91845703125  |0.7 |



## Mixup
![image](https://github.com/unsky/mixup/blob/master/1.png)
## Mixup in feature map （resnet50 head conv）
![image](https://github.com/unsky/mixup/blob/master/5.png)

## ERM
![image](https://github.com/unsky/mixup/blob/master/2.png)

# Usage
install mxnet0.12
The mixup is in:symbols/mixup.py
you can use it in your codes like:

```
data ,label = mx.sym.Custom(data= data,label = label,alpha = 0.2,num_classes = num_classes,batch_size = batch_size,mix_rate =0.7,op_type = 'MixUp')
```
label is the vector like [4,8,...9]
### download the dataset
http://data.mxnet.io/data/cifar10/cifar10_val.rec

http://data.mxnet.io/data/cifar10/cifar10_train.rec



### train:
```
./train.sh
```
### test:
```
./test.sh
```
