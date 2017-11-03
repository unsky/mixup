# mixup
mixup: Beyond Empirical Risk Minimization
This is an implement and Improvement  on mixup: Beyond Empirical Risk Minimization https://arxiv.org/abs/1710.09412

# The improvement 

1. add backward
2. add mix rate


two scenes:
![image](https://github.com/unsky/mixup/blob/master/3.png)


the detail design of mixUPlayer:

![image](https://github.com/unsky/mixup/blob/master/4.png)


# The results:
the results is not very strict，i will update it  when i have a strict base line.

|                        | alpha         | mix_rate  | AP |
| -------------          |:-------------:| -----:      | -----:   |
| (ERM)resnet50 90epoch  |      -        |-            | 0.77     |
| (mixup)resnet50 90epoch|      0.2     |0.7           | 0.78     |
| (mixup)resnet50 200epoch|      0.2     |0.7          | 0.83     |
| (mixup in feature maps)|      0.2     |0.7          | todo     |

## mixup:
![image](https://github.com/unsky/mixup/blob/master/1.png)

## ERM
![image](https://github.com/unsky/mixup/blob/master/2.png)

# usage
install mxnet0.12
The mixup is in:symbols/mixup.py
you can use it in your codes like:

```
data ,label = mx.sym.Custom(data= data,label = label,alpha = 0.2,num_classes = num_classes,batch_size = batch_size,mix_rate =0.7,op_type = 'MixUp')
```
label is the vector like [4,8,...9]


train:
```
./train.sh
```
test:
```
./test
```
