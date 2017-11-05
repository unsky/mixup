#!/usr/bin/env python

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

"""mixup layer for MXNet.
"""
import os

import numpy as np
import mxnet as mx
from mxnet import autograd


class MixUp(mx.operator.CustomOp):
    def __init__(self,  alpha,num_classes,batch_size,mix_rate):
        super(MixUp, self).__init__()
        self.alpha = alpha 
        self.mix_rate = mix_rate
	self.num_classes = num_classes
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()  # 
   	mix_len = int(self.mix_rate*data.shape[0])
 	if (mix_len%2)!=0:
	    mix_len = mix_len+1
        unmix_data = data[mix_len:,:,:,:]
        data = data[:mix_len,:,:,:]
 

        
        y = in_data[1].asnumpy().astype('int')
        unmix_y = y[mix_len:]
	y = y[:mix_len]

        length = data.shape[0] / 2

        label = np.zeros((data.shape[0],self.num_classes))
        batch_size = y.shape[0]
        label[np.arange(label.shape[0],dtype = 'int'),y] = 1
        self.lamdba1 = np.random.beta(self.alpha,self.alpha,size = batch_size/2)
        self.lamdba2 = np.random.beta(self.alpha,self.alpha,size = batch_size/2)
        self.data_lamdba1 =mx.nd.array( self.lamdba1.reshape(-1,1,1,1))
        self.data_lamdba2 = mx.nd.array(self.lamdba2.reshape(-1,1,1,1))
        self.label_lamdba1 =mx.nd.array( self.lamdba1.reshape(-1,1))
        self.label_lamdba2 = mx.nd.array(self.lamdba2.reshape(-1,1))
       
        data1 = data[:length,:,:,:]
        data2 = data[length :,: ]
        y1 = label[:length,:]
        y2 = label[length:,:]
        data1 = mx.nd.array(data1)
        data2 = mx.nd.array(data2)
        y1 = mx.nd.array(y1)
        y2 = mx.nd.array(y2)        
        data, label= self.mix(data1,data2,y1,y2,self.data_lamdba1,self.data_lamdba2,self.label_lamdba1,self.label_lamdba2)
#	data -= mx.nd.mean(data, axis = 0) # zero-center
#	data /= mx.nd.norm(data, axis = 0) # normalize
	unmix_label = np.zeros((unmix_y.shape[0],self.num_classes))
	unmix_label[np.arange(unmix_label.shape[0],dtype = 'int'),unmix_y] = 1


        data = mx.nd.concatenate([data,mx.nd.array(unmix_data)],axis=0)
        label = mx.nd.concatenate([label,mx.nd.array(unmix_label)],axis=0)

        
        self.assign(out_data[0],req[0],data)
	self.assign(out_data[1],req[0],label)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.app_backward(req, out_grad, in_data, out_data, in_grad, aux)

    def app_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        data = in_data[0].asnumpy()  
        mix_len = int(self.mix_rate*data.shape[0])
        if (mix_len%2)!=0:
            mix_len = mix_len+1
        unmix_data = data[mix_len:,:,:,:]
        data = data[:mix_len,:,:,:]

        length = data.shape[0] / 2

        data1 = data[:length,:,:,:]
        data2 = data[length :,: ]
        data1 = mx.nd.array(data1)
        data2 = mx.nd.array(data2)


        data1.attach_grad()
	data2.attach_grad()
	with autograd.record():
		data= self.dif_mix(data1,data2,self.data_lamdba1)
        data.backward()
        diff1_data1 = data1.grad
	diff1_data2 = data2.grad
        data1.attach_grad()
        data2.attach_grad()
        with autograd.record():
                data= self.dif_mix(data1,data2,self.data_lamdba2)
        data.backward()
        diff2_data1 = data1.grad
        diff2_data2 = data2.grad
        diff_data1 = diff1_data1 + diff2_data1
        diff_data2 = diff1_data2 + diff2_data2
        unmix_grad = mx.nd.ones(shape = unmix_data.shape)


        grad = mx.nd.concatenate([diff_data1,diff_data2,unmix_grad],axis=0)
        self.assign(in_grad[0], req[0], grad*out_grad[0])
	self.assign(in_grad[1], req[0],0)


    def mix(self, data1, data2,y1,y2,data_lamdba1,data_lamdba2,label_lamdba1,label_lamdba2):
       
        data_1 = data_lamdba1 * data1 + (1-data_lamdba1)*data2
        label_1 = label_lamdba1*y1 + (1-label_lamdba1)*y2
     
  	data_2 = data_lamdba2*data1 + (1-data_lamdba2)*data2
   	label_2 = label_lamdba2*y1 + (1-label_lamdba2)*y2
	data = mx.nd.concatenate([data_1,data_2],axis=0)
	label = mx.nd.concatenate([label_1,label_2],axis=0)	
	return data,label
    def dif_mix(self,data1,data2,data_lamdba):
	data  =  data_lamdba * data1 + (1-data_lamdba)*data2
	return data



@mx.operator.register("MixUp")
class MixUpProp(mx.operator.CustomOpProp):
    def __init__(self,alpha,num_classes,batch_size,mix_rate):
        super(MixUpProp, self).__init__(need_top_grad=True)
	self._alpha = float(alpha)
	self._num_classes = int(num_classes)
	self._batch_size = int(batch_size)
        self._mix_rate = float(mix_rate)

    def list_arguments(self):

        return ['data','label']

    def list_outputs(self):
        return ['mixed_data','label']

    def create_operator(self, ctx, shapes, dtypes):
        return MixUp(self._alpha,self._num_classes,self._batch_size,self._mix_rate)

    def infer_shape(self, in_shape):
       
 
        return in_shape, [in_shape[0],[in_shape[1][0],self._num_classes]]

