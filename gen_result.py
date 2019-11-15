
from numpy import *
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import matplotlib.pyplot as plt
CLS_NUM = 200
POS_NUM = 2
LABEL_VALUE_NUM = 9

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap = plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = array(range(LABEL_VALUE_NUM))
    plt.xticks(xlocations, labels, rotation=45)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


res = []
with open('./rank-00000') as f:
    for line in f:
        pred = int(line.strip(',\r\n;'))
        res.append(pred)

file_name = open('./data/pred.list').read().strip('\r\n')
data = sio.loadmat(file_name)
label = data['label']
Class_num = data['Class_num1']
Class_num = Class_num.tolist()[0]
class_num_new = sorted(Class_num,reverse=True)[0:9]

LABEL = mat( zeros( (1,1) ) )
for  i in range(LABEL_VALUE_NUM) :
    num = class_num_new[i]
    LABEL = hstack((LABEL , i*mat( ones( (1,num) ) )))
LABEL = LABEL.tolist()[0]
LABEL = map(int, (LABEL[1:]))
Correct = mat( zeros( (LABEL_VALUE_NUM,LABEL_VALUE_NUM) ) )
cm =  confusion_matrix(LABEL, res)
right_num =0;
for i in range(LABEL_VALUE_NUM):
    right_num = right_num + cm[i,i]
Accuracy = 1.0*right_num/( sum (class_num_new[:LABEL_VALUE_NUM]))
print "Accuracy:", Accuracy
'''
plt.style.use('ggplot')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()
'''
set_printoptions(precision=2)
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, newaxis]
print cm_normalized
plt.figure()
for x_val in range(LABEL_VALUE_NUM):
    for y_val in range(LABEL_VALUE_NUM):
        c = cm_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
#offset the tick
tick_marks = array(range(LABEL_VALUE_NUM)) + 0.5
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, class_num_new[:LABEL_VALUE_NUM], title='Normalized confusion matrix')
#show confusion matrix
plt.show()
