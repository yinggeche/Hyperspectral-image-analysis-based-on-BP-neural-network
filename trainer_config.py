
from paddle.trainer_config_helpers import *
import numpy as np
################################### DATA Configuration #############################################
is_predict = get_config_arg('is_predict', bool, False)
trn = './data/train.list' if not is_predict else None
tst = './data/test.list' if not is_predict else './data/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(
    train_list=trn, test_list=tst, module="dataprovider", obj=process)
################################### Parameter Configuaration #######################################

CLS_NUM = 200
POS_NUM = 2
LABEL_VALUE_NUM = 9

hidden_size1 =  80
hidden_size2 =  20
hidden_dim = 25
batch_size = 64 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=5e-3, #TODO
    regularization=L2Regularization(batch_size * 8e-4),
    learning_method=AdamOptimizer()) #RMSPropOptimizer()) #TODO ())AdaGradOptimizer
################################### Algorithm Configuration ########################################
output_label = []
forinput = []
CLS = data_layer(name='CLS', size=CLS_NUM)
POS = data_layer(name='POS', size=POS_NUM)
forinput = [CLS, POS]
'''
link_param = ParamAttr(
    name=/ '_link_vec.w', initial_max=1.0, initial_min=-1.0)
link_vec = fc_layer(input=link_encode, size=emb_size, param_attr=link_param)
score = fc_layer(input=link_vec, size=4, act=SoftmaxActivation())
'''
link_param1 = ParamAttr(
        name='_par.w1', initial_max=1.0, initial_min=-1.0)
link_param2 = ParamAttr(
        name='_par.w2', initial_max=1.0, initial_min=-1.0)
#CLS_emd = embedding_layer(input=CLS, size=emb_size, param_attr=link_param)
hidden1 = fc_layer(name='CLS_hidden', input=CLS, size=hidden_size1, act=STanhActivation(), param_attr=link_param1)
hidden3 = fc_layer(name='POS_hidden', input=POS, size=hidden_size2, act=SigmoidActivation(), param_attr=link_param2)
hidden2 = mixed_layer(
    name='mixed',
    size=hidden_dim,
    act=STanhActivation(),
    #act=SigmoidActivation(),
    bias_attr=True,
    input=[
    #    full_matrix_projection(CLS_emd),
        full_matrix_projection(hidden1),
        full_matrix_projection(hidden3),
        ])
score = fc_layer(
    name='score',
    size=LABEL_VALUE_NUM,
    act=SoftmaxActivation(),
    bias_attr=False,
    input=hidden2,)

if is_predict:
    maxid = maxid_layer(score)
    output_label.append(maxid)
else:
    label = data_layer(name='label' , size=LABEL_VALUE_NUM )
    forinput.append(label)
    cls = multi_binary_label_cross_entropy(
        name='cost',
        input=score, label=label)
        #input=score, name="cost_%dmin" % ((i + 1) * 5), label=label)
    output_label.append(cls)
'''
        if i == 0:
            output_label = cls
        else:
            output_label = addto_layer(input=[cls,output_label], name = "sum_%d" %i, act=ReluActivation(), bias_attr=False)
'''
inputs(forinput)
outputs(output_label)
