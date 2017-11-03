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

"""sparse softmax-ross-entropy loss layer for MXNet.
"""
import os

import numpy as np
import mxnet as mx
from mxnet import autograd

# ref: http://mxnet.io/how_to/new_op.html

class SparseSoftmaxCrossEntropyLoss(mx.operator.CustomOp):


    def forward(self, is_train, req, in_data, out_data, aux):
	X = in_data[0][:]
	y = in_data[1]
	p = mx.nd.SoftmaxActivation(X)
#	print p

	loss = -1*mx.nd.log(mx.nd.sum(y*p,axis =1))
#        print y 
        
            # Just copy the predictions forward
        self.assign(out_data[0], req[0], loss)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.approx_backward(req, out_grad, in_data, out_data, in_grad, aux)
        #self.exact_backward(req, out_grad, in_data, out_data, in_grad, aux)

    def approx_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        X = in_data[0][:] # shape=(b,d)
        y = in_data[1]
        X.attach_grad()
 	with autograd.record():
	     p = mx.ndarray.SoftmaxActivation(X)
	     
	     loss = -1*mx.nd.log(mx.nd.sum(y*p,axis =1))
        loss.backward()
	#print X.grad/X.shape[0]
        self.assign(in_grad[0], req[0], X.grad/X.shape[0])


@mx.operator.register("SparseSoftmaxCrossEntropyLoss")
class SparseSoftmaxCrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SparseSoftmaxCrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['preds']

    def create_operator(self, ctx, shapes, dtypes):
        return SparseSoftmaxCrossEntropyLoss()

    def infer_shape(self, in_shape):
        
        output_shape = in_shape[0]
        return in_shape, [[output_shape[0],]]
