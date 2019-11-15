
from paddle.trainer.PyDataProvider2 import *
import sys
from numpy import *
import scipy.io as sio
CLS_NUM = 200
POS_NUM = 2
LABEL_VALUE_NUM = 9


def initHook(settings, file_list, **kwargs):
    del kwargs  #unused

    settings.pool_size = sys.maxint

    settings.input_types =[dense_vector(CLS_NUM) ,
                           dense_vector(POS_NUM) ,
                           integer_value_sequence(LABEL_VALUE_NUM) ,
                            ]
                            #integer_value_sequence(TERM_NUM) ,
                            #[dense_vector(TERM_NUM)]


@provider(
    init_hook=initHook, cache=CacheType.CACHE_PASS_IN_MEM, should_shuffle=True)
def process(settings, file_name):
    data = sio.loadmat(file_name)
    Class_num = data['Class_num1']
    clsAll = data['clsAll']
    posAll = data['posAll']
    Class_num = Class_num.tolist()[0]
    class_num_new = sorted(Class_num,reverse=True)
    CLS = mat( zeros( (CLS_NUM,1) ) )
    POS = mat( zeros( (POS_NUM,1) ) )
    LABEL = mat( zeros( (1,1) ) )
    num = 0;
    m = 0;
    for  i in range(LABEL_VALUE_NUM) :
        num = class_num_new[i]
        m = m + num
        pos = Class_num.index( num )
        beforenum = sum( Class_num[0:pos] );
        CLS = hstack((CLS,clsAll[:,beforenum:beforenum + num ]))
        POS = hstack((POS ,posAll[:,beforenum:beforenum + num ]))
        LABEL = hstack((LABEL , i*mat( ones( (1,num) ) )))
    for i in range(1,m+1):
        yield [CLS[:,i].T.tolist()[0], POS[:,i].T.tolist()[0],  map(int, (LABEL[:,i].T.tolist()[0])) ]


def predict_initHook(settings, file_list, **kwargs):
        del kwargs  #unused

        settings.pool_size = sys.maxint
        #Use a time seires of the past as feature.
        #Dense_vector's expression form is [float,float,...,float]

        settings.input_types =[dense_vector(CLS_NUM) ,
                               dense_vector(POS_NUM) ,]


@provider(init_hook=predict_initHook, should_shuffle=False)
def process_predict(settings, file_name):
    data = sio.loadmat(file_name)
    Class_num = data['Class_num1']
    clsAll = data['clsAll']
    posAll = data['posAll']
    Class_num = Class_num.tolist()[0]
    class_num_new = sorted(Class_num,reverse=True)
    CLS = mat( zeros( (CLS_NUM,1) ) )
    POS = mat( zeros( (POS_NUM,1) ) )
    LABEL = mat( zeros( (1,1) ) )
    num = 0;
    m = 0;
    for  i in range(LABEL_VALUE_NUM) :
        num = class_num_new[i]
        m = m + num
        pos = Class_num.index( num )
        beforenum = sum( Class_num[0:pos] );
        CLS = hstack((CLS,clsAll[:,beforenum:beforenum + num ]))
        POS = hstack((POS ,posAll[:,beforenum:beforenum + num ]))
        LABEL = hstack((LABEL , i*mat( ones( (1,num) ) )))
    for i in range(1,m+1):
        yield [CLS[:,i].T.tolist()[0], POS[:,i].T.tolist()[0] ]
