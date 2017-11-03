import mxnet as mx
import numpy as np
class LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(LossMetric, self).__init__('Loss')

    def update(self, labels, preds):

        pred = preds[0].asnumpy()

    

        self.sum_metric +=np.sum(pred)
        self.num_inst += len(pred)

class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(AccMetric, self).__init__('Acc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy()

        pred = np.argmax(pred,axis=1).astype('int')
        labels = labels[0].asnumpy().astype('int')
#  	print pred,labels
       
        print np.sum(labels==pred)/(float(labels.shape[0]))
        self.sum_metric +=np.sum(pred==labels)
        self.num_inst += len(pred)
