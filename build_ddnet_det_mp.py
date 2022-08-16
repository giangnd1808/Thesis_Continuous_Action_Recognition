import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import random
from tqdm import tqdm
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from tensorflow.keras.callbacks import *
from keras.layers.convolutional import *
import tensorflow as tf

#Initialize the setting
random.seed(1234)

class Config():
    def __init__(self):
        self.frame_l = 16 # the length of frames
        self.joint_n = 33 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_num = 2 # the number of class, (= 8 if using subsets)
        self.feat_d = 528
        self.filters = 32
        self.nd = 60

#Define data processing functions
def zoom(p,target_l=16,joints_num=33,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new

def sampling_frame(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]    
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
    return p

from scipy.spatial.distance import cdist
def get_CG(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l):
        #distance max 
        d_m = cdist(p[f],np.concatenate([p[f],np.zeros([1,C.joint_d])]),'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M)   
    return M

def norm_train(p):
    # normolize to start point, use the center for hand case
    # p[:,:,0] = p[:,:,0]-p[:,3:4,0]
    # p[:,:,1] = p[:,:,1]-p[:,3:4,1]
    # p[:,:,2] = p[:,:,2]-p[:,3:4,2]
    # # return p
       
    p[:,:,0] = p[:,:,0]-np.mean(p[:,:,0])
    p[:,:,1] = p[:,:,1]-np.mean(p[:,:,1])
    p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p
def norm_train2d(p):
    # normolize to start point, use the center for hand case
    # p[:,:,0] = p[:,:,0]-p[:,3:4,0]
    # p[:,:,1] = p[:,:,1]-p[:,3:4,1]
    # p[:,:,2] = p[:,:,2]-p[:,3:4,2]
    # # return p
       
    p[:,:,0] = p[:,:,0]-np.mean(p[:,:,0])
    p[:,:,1] = p[:,:,1]-np.mean(p[:,:,1])
    # p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p

#Building the model
def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W]) 
    return x
def poses_diff_2(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    # x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W]) 
    return x
def pose_motion_2(D, frame_l):
    x_1 = Lambda(lambda x: poses_diff_2(x))(D)
    x_1 = Reshape((frame_l,-1))(x_1)
    return x_1

def pose_motion(P,frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    x_1 = Reshape((frame_l,-1))(P)
    return P_diff_slow,P_diff_fast
# def reshape_x_2(D, frame_l):
#     x_1 = Lambda(lambda y: poses_diff_2(y))(D)
#     x_1 = Reshape((frame_l, -1))(D)

def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=20,joint_d=3,feat_d=190,filters=32, nd=60):
    drop_rate = 0.1
    M = Input(shape=(frame_l,feat_d))
    P = Input(shape=(frame_l,joint_n,joint_d))
    # D = Input(shape =(frame_l, joint_n, joint_d))
    # x_ = pose_motion_2(D, frame_l)
    diff_slow,diff_fast = pose_motion(P,frame_l)
    


    x = c1D(M,filters*2,1)
    x = SpatialDropout1D(drop_rate)(x)
    x = c1D(x,filters,3)
    x = SpatialDropout1D(drop_rate)(x)
    x = c1D(x,filters,1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(drop_rate)(x)

    
    # x_1 = c1D(x_1, filters*2,1)
    # x_1 = SpatialDropout1D(drop_rate)(x_1)
    # x_1 = c1D(x_1, filters, 3)
    # x_1 = SpatialDropout1D(drop_rate)(x_1)
    # x_1 = c1D(x_1, filters,1)
    # x_1 = MaxPooling1D(2)(x_1)
    # x_1 = SpatialDropout1D(drop_rate)(x_1)

    x_d_slow = c1D(diff_slow,filters*2,1)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)

    # x = c1D(diff_fast,filters*2,1)
    # x = SpatialDropout1D(drop_rate)(x)
    # x = c1D(x,filters,3) 
    # x = SpatialDropout1D(drop_rate)(x)
    # x = c1D(x,filters,1) 
    # x = SpatialDropout1D(drop_rate)(x)

    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3) 
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1) 
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
   
    x = concatenate([x,x_d_slow,x_d_fast])
    x = block(x,filters*2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(drop_rate)(x)
    
    x = block(x,filters*4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(drop_rate)(x)

    x = block(x,filters*8)
    x = SpatialDropout1D(drop_rate)(x)
    
    return Model(inputs=[M,P],outputs=x)


def build_DD_Net_det(C):
    M = Input(name='M', shape=(C.frame_l,C.feat_d))  
    P = Input(name='P', shape=(C.frame_l,C.joint_n,C.joint_d)) 
    # D = Input(name ='D', shape =(C.frame_l, C.joint_n,C.joint_d))
    FM = build_FM(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.filters)
    
    x = FM([M,P])

    x = GlobalMaxPool1D()(x)
    
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    ######################Self-supervised part
    model = Model(inputs=[M,P],outputs=x)
    return model


#Data processing
def kinect2mp_spec_joint(mp, joint1, joint2):
    kinect = np.zeros(3, dtype=np.float32)
    kinect[0] = (mp[joint1][0] + mp[joint2][0]) / 2
    kinect[1] = (mp[joint1][1] + mp[joint2][1]) / 2
    kinect[2] = (mp[joint1][2] + mp[joint2][2]) / 2
    return kinect

def kinect2mp(mp):
    kinect2mp_list = [[3,0], [4,11], [5,13], [6,15], [8,12], [9,14], [10,16], [12,23],
    [13,25], [14,27], [15,31], [16,24], [17,26], [18,28], [19,32]]
    kinect = np.zeros((20,3), dtype=np.float32)
    for jointID in kinect2mp_list:
        kinect[jointID[0]] = mp[jointID[1]]
    kinect[0] = kinect2mp_spec_joint(mp, 23, 24)
    kinect[2] = kinect2mp_spec_joint(mp, 11, 12)
    kinect[1] = kinect2mp_spec_joint(kinect, 0, 2)
    kinect[11] = kinect2mp_spec_joint(mp, 18, 20)
    kinect[7] = kinect2mp_spec_joint(mp, 17, 19)
    return kinect

def data_generator_test(T,C):
    p = np.copy(T)
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    p = norm_train(p)
    M = get_CG(p,C)
    f1 = np.expand_dims(M, axis = 0)
    f2 = np.expand_dims(p, axis = 0)

    return f1,f2
