import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import cv2
 
 
 
#处理的12X12网络
def Pnet():
    input = tf.keras.Input(shape=[None, None, 3])
    x = tf.keras.layers.Conv2D(10, (3, 3), name='conv1',kernel_regularizer=keras.regularizers.l2(0.0005))(input)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3),name='conv2',kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3),name='conv3',kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU3')(x)
    classifier = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax',name='conv4-1')(x)
    #如果input 是大于12*12,[1,2]不为1
    #cls_prob = tf.squeeze(classifier, [1, 2], name='cls_prob')
    bbox_regress = tf.keras.layers.Conv2D(4, (1, 1), name='conv4-2')(x)
    #bbox_pred = tf.squeeze(bbox_regress,[1,2],name='bbox_pred')
    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model
 
#处理的24X24网络
def Rnet():
    """定义RNet网络的架构"""
    input = tf.keras.Input(shape=[24, 24, 3])
    x = tf.keras.layers.Conv2D(28, (3, 3),strides=1,padding='valid',name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu1')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.Conv2D(48, (3, 3),strides=1,padding='valid',name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (2, 2),strides=1,padding='valid',name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu3')(x)
    x = tf.keras.layers.Permute((3, 2, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, name='conv4')(x)
    x = tf.keras.layers.PReLU(name='prelu4')(x)
    classifier = tf.keras.layers.Dense(2,activation='softmax',name='conv5-1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='conv5-2')(x)
    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model
 
#处理的48X48网络
def Onet():
    """定义ONet网络的架构"""
    input = tf.keras.layers.Input(shape=[48, 48, 3])
    # 48,48,3 -> 23,23,32
    x = tf.keras.layers.Conv2D(32, (3, 3),strides=1,padding='valid',name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = tf.keras.layers.Conv2D(64, (3, 3),strides=1,padding='valid',name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = tf.keras.layers.Conv2D(64, (3, 3),strides=1,padding='valid',name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu3')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = tf.keras.layers.Conv2D(128, (2, 2),strides=1,padding='valid',name='conv4')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = tf.keras.layers.Permute((3, 2, 1))(x)
    # 1152 -> 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, name='conv5')(x)
    x = tf.keras.layers.PReLU(name='prelu5')(x)
    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = tf.keras.layers.Dense(2,activation='softmax',name='conv6-1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='conv6-2')(x)
    #landmark_regress = tf.keras.layers.Dense(10, name='conv6-3')(x)
    #model = tf.keras.models.Model([input], [classifier, bbox_regress,landmark_regress])
    #my code
    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model
 
 
 
#人脸分类损失函数
def cls_ohem(cls_prob, label):
 
    zeros = tf.zeros_like(label, dtype=tf.float32)
    # 若label中的值小于等于0，则为0，否则为1，就是把label中-1变为0
    label_filter_invalid = tf.where(tf.math.less(label,[0]),zeros,label)
 
    ## 类别size[2*batch]
    num_cls_prob = tf.size(cls_prob)
 
    #把cls_porob变成一维
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,dtype=tf.int32)
    num_row = tf.cast(cls_prob.get_shape()[0],dtype=tf.int32)  #[batch]
 
    # 对应某一batch而言，batch*2为非人类别概率，
    # batch*2+1为人概率类别,indices为对应 cls_prob_reshpae
    # 应该的真实值，后续用交叉熵计算损失
    row = tf.range(num_row)*2   #[0 2 4 6]
    #就是如果label是pos就看1X2中的第2个，neg或part就看第1个
    indices_ = row + label_int
    # 从cls_prob_reshape中获取 索引为indices_的值，squeeze后变成一维的长度为batch_size的张量。
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    #OHEM向前时，全部的Roi通过网络
    loss = -tf.math.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
 
    # 把标签为±1的样本对应的索引设为1，其余设为0 #这一步是用来计算较大的候选RIO 用来OHEM
    valid_inds = tf.where(label < zeros,zeros,ones)
    #获取有效的样本数(即标签为±1  (正样本和负样本的数量)
    num_valid = tf.reduce_sum(valid_inds)
 
    #num_keep_radio = 0.7  选取70%的数据
    keep_num = tf.cast(num_valid*0.7,dtype=tf.int32)
    # print("keep_num",keep_num)
 
    # 只选取neg，pos的70%损失
    #loss = loss * num_valid
    loss = loss * valid_inds
 
    #OHEM就是对loss从高到底排序
    # 反向时，根据排序选择Batch-size/N 个loss值得最大样本来后向传播model的权重
    loss,_ = tf.math.top_k(loss, k=keep_num)
 
    return tf.math.reduce_mean(loss)
 
 
# 人脸框损失函数
def bbox_ohem(bbox_pred,bbox_target,label):
 
    zeros_index = tf.zeros_like(label,dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
 
    # 等于±1的有效为1，不等于1的无效为0，即筛选出pos和part的索引-OHEM策略
    valid_inds = tf.where(tf.math.equal(tf.math.abs(label),1),ones_index,zeros_index)
 
    #计算平方差损失
    square_error = tf.math.square(bbox_pred - bbox_target)  #16-1-16-14
    square_error = tf.math.reduce_sum(square_error,axis=1)  #16*16*4
 
 
    # 保留数据的个数
    num_valid = tf.math.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid,dtype=tf.int32)
 
 
    #OHEM策略，保留部分pos,part的损失
    #square_error = square_error * num_valid
    square_error = square_error * valid_inds
 
    # 选出最大的进行反向传播
    _,k_index = tf.math.top_k(square_error,k=keep_num)
    # 将部分pos样本和part样本的平方和提取出来
    square_error = tf.gather(square_error, k_index)
    
    print('square_error',square_error)
    
    if np.isnan(tf.reduce_mean(square_error)):
        return tf.zeros(0)
 
    return tf.reduce_mean(square_error)
 
 
#人脸五官损失函数
def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
 
    # 只保留landmark数据
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
 
    # 计算平方差损失
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
 
    # 保留数据个数
    num_valid = tf.math.reduce_sum(valid_inds) # 0
    keep_num = tf.cast(num_valid, dtype=tf.int32) # 0
 
    # 保留landmark部分数据损失
    square_error = square_error*valid_inds
    square_error, k_index = tf.nn.top_k(square_error, k=keep_num)
    # square_error = tf.gather(square_error, k_index)
 
    return tf.math.reduce_mean(square_error) # 当square_error为空时会出现nan bug
 
 
#准确率
def cal_accuracy(cls_prob,label):
 
    # 预测最大概率的类别，0代表无人，1代表有人
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
 
    #返回pos和neg示例的索引 :按元素返回（x> = y）的真值
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    #true_label选出picked(pos和neg)坐标
    label_picked = tf.gather(label_int,picked)
    #pre_label选出picked(pos和neg)坐标
    pred_picked = tf.gather(pred,picked)
 
    # accuracy_op = tf.math.reduce_sum(tf.cast(tf.equal(label_picked,pred_picked),dtype=tf.float32))
    # accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(label_picked, pred_picked), tf.float32))
    return label_picked,pred_picked
    # return accuracy
