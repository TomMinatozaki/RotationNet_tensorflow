import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
os.environ['CUDA_VIDISIBLE_DEVICES'] = '3'

def tf_repeat(y,repeat_num):
        return tf.reshape(tf.tile(tf.expand_dims(y,axis=-1),[1,repeat_num]),[-1])\
def read_png(img_dir_list,cls='train'):
    imgs_list = []
    label_list = []
    for item_list in img_dir_list:
        marker = item_list[0][:-13]
        imgs   = np.array([cv2.imread(os.path.join(DATA_DIR,cls,marker,item)) for item in item_list])
        imgs_list.append(imgs)
        label_list.append(cls_dir[marker])
    print(os.path.join(DATA_DIR,cls,marker,item))
    return np.array(imgs_list),np.array(label_list)
vcand = np.load('/data1/home/tom/modelnet40/pytorch-rotationnet/vcand_case2.npy')
#from resnet_Helper import * 
#from vgg_Helper import *
from Inception_Helper import *
#tf.enable_eager_execution()
img_size = 224
nview   = 20
num_cls = 40+1 
n_samp  = 3
batch_size = nview*n_samp
X = tf.placeholder(shape=[n_samp,nview,img_size,img_size,3] , dtype=tf.float32,name = 'X')
Y = tf.placeholder(shape=[n_samp],dtype=tf.int32,name='Y')
img_dir = np.load('./good_img_dir.npy')
item = img_dir[0]
#os.chdir('../ModelNet40/train/airplane/')

feature,end_points = vgg_16(X_4d,num_classes=nview*num_cls)
output  = tf.reshape(feature,(n_samp,nview*nview,num_cls))
softmax = tf.nn.log_softmax(output,axis = -1)
output_ = softmax[:,:,1:] - tf.tile(softmax[:,:,:1],[1,1,num_cls-1])
target_  = tf.zeros([batch_size*nview],dtype=tf.int64)
scores   = tf.zeros((n_samp,vcand.shape[0],num_cls-1))
j_box = []
for j in range(vcand.shape[0]):
    j_box.append([])
    for k in range(vcand.shape[1]):
        j_box[-1].append(vcand[j][k]*nview+k)
j_box = np.array(j_box,dtype=np.int32)
j_box_where = np.zeros([nview*nview,60])
for j,box in enumerate(j_box):
    j_box_where[:,j][box]=1
j_box_where = np.expand_dims(j_box_where,axis=0)
j_box_where = np.expand_dims(j_box_where,axis=2)
j_box_where = np.tile(j_box_where,[n_samp,1,num_cls-1,1])
output_expand = tf.expand_dims(output_,axis=-1)
output_expand = tf.tile(output_expand, [1,1,1,60])
output_expand_zeros = tf.zeros_like(output_expand)
scores = tf.where(j_box_where,output_expand,output_expand_zeros)
scores = tf.reduce_sum(scores,axis=1)
j_max       = tf.argmax(tf.gather_nd(scores,tf.stack((tf.range(n_samp),Y),axis=1)),axis=-1)
j_max_index = tf.gather(j_box,j_max)
j_max_index = tf.cast(j_max_index,tf.int64)
j_max_index_flat = tf.reshape(j_max_index,[-1])
j_max_indes_for_sparse = tf.stack( (np.repeat(np.arange(n_samp),20) , j_max_index_flat),axis=1)
delta = tf.SparseTensor(indices=j_max_indes_for_sparse,values=tf_repeat(Y,20),dense_shape=[n_samp,nview*nview])
target_     = tf.sparse_tensor_to_dense(delta,validate_indices=False)
loss        = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=target_)
optim = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


