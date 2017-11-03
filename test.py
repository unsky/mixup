# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
test pretrained models
"""
from __future__ import print_function
import mxnet as mx
from common import find_mxnet, modelzoo,metric
from score import score
from symbols import sparse_softmax,mixup,softmax

def test_mixup(**kwargs):
    from importlib import import_module
    net = import_module('symbols.resnet_mixup')
    sym = net.get_symbol(num_classes = 10, num_layers=50, image_shape='3,28,28', conv_workspace=256,batch_size =256,is_train =False)
    acc = metric.AccMetric()
  
    (speed,) = score(sym=sym, prefix = 'models/mix',epoch =90, data_val='data/cifar10_val.rec', rgb_mean='123.68,116.779,103.939', metrics=acc,gpus='0', batch_size=256)
   
 
if __name__ == '__main__':
    gpus = mx.test_utils.list_gpus()
    assert len(gpus) > 0
    batch_size = 16 * len(gpus)
    gpus = ','.join([str(i) for i in gpus])

    kwargs = {'gpus':gpus, 'batch_size':batch_size, 'max_num_examples':500}
    test_mixup(**kwargs)
