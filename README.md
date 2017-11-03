# mixup
mixup: Beyond Empirical Risk Minimization
This is an implement and Improvement  on mixup: Beyond Empirical Risk Minimization https://arxiv.org/abs/1710.09412

# The improvement 








roughly
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

##
![image](https://github.com/unsky/mixup/blob/master/2.png)
