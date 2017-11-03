export export PYTHONPATH=/root/cs/mxnet/python:$PYTHONPATH
python train_cifar10.py --gpus 0,1 --model-prefix  'models/mix'
#python test_score.py
