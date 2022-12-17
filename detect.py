import tensorflow as tf
import numpy as np
import cv2
from mtcnn_model import Pnet,Rnet,Onet


min_face_size = 20 
 
#预测处理数据
def processed_img(img, scale):
    '''预处理数据，转化图像尺度并对像素归一到[-1,1]
    '''
    h,w,_ = img.shape
    n_h = int(h*scale)
    n_w = int(w*scale)
    dsize = (n_w,n_h)
    img_resized = cv2.resize(np.array(img), dsize,interpolation=cv2.INTER_LINEAR)
    img_resized = (img_resized - 127.5)/128
    return img_resized
 
# 生成边框
def generate_bounding_box(cls_pro,bbox_pred,scale,threshold):
 
    stride = 2
    cellsize = 12
    # softmax layer 1 for face, return a tuple with an array of row idxs and
    # an array of col idxs
    # locate face above threshold from cls_map
    t_index = np.where(cls_pro > threshold)
 
    # find nothing
    if t_index[0].size == 0:
        return np.array([])
        # 偏移量
    bbox_pred = bbox_pred[t_index[0], t_index[1], :]
    bbox_pred = np.reshape(bbox_pred, (-1, 4))
    score = cls_pro[t_index[0], t_index[1]]
    score = np.reshape(score, (-1, 1))
 
    x1Arr = np.round((stride * t_index[1]) / scale)
    x1Arr = np.reshape(x1Arr, (-1, 1))
    y1Arr = np.round((stride * t_index[0]) / scale)
    y1Arr = np.reshape(y1Arr, (-1, 1))
    x2Arr = np.round((stride * t_index[1] + cellsize) / scale)
    x2Arr = np.reshape(x2Arr, (-1, 1))
    y2Arr = np.round((stride * t_index[0] + cellsize) / scale)
    y2Arr = np.reshape(y2Arr, (-1, 1))
 
    bboxes = np.concatenate([x1Arr, y1Arr, x2Arr, y2Arr, score, bbox_pred], -1)
 
 
    return bboxes
 
#校正边框
def calibrate_box(bboxes,offsets):
    """
 
    :param bboxes: [n,5]
    :param offsets: [n,4]
    :return: [n,5]
    """
 
    x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
 
    w = np.expand_dims(w,1)
    h = np.expand_dims(h,1)
 
    translation = np.hstack([w,h,w,h]) * offsets
    bboxes[:,0:4] = bboxes[:,0:4] + translation
 
    return bboxes
 
# 非极大抑制
def nms(dets, thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
 
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
 
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
 
        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
 
    return keep
 
#矩形转正方形
def convert_to_square(bboxes):
    """
    将边框转换成正方形
    :param bboxes: [n,5]
    :return:
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
 
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes
 
#处理超出范围的边框
def pad(bboxes, w, h):
    '''将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : 为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ey, ex : 调整后的box在原图上右下角的坐标
    '''
    tw, th = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    n_box = bboxes.shape[0]
 
    dx, dy = np.zeros((n_box,)), np.zeros((n_box,))
    edx, edy = tw.copy() - 1, th.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
 
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tw[tmp_index]  - 1 - (ex[tmp_index] - w + 1)
    ex[tmp_index] = w - 1
 
    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = th[tmp_index] - 1 - (ey[tmp_index] - h + 1)
    ey[tmp_index] = h - 1
 
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0
 
    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0
 
    return_list = [dy, edy, dx, edx, y, ey, x, ex, tw, th]
    return_list = [item.astype(np.int32) for item in return_list]
 
    return return_list
 
    #return x.astype(np.int32), y.astype(np.int32), ex.astype(np.int32), ey.astype(np.int32), dx.astype(np.int32), dy.astype(np.int32), edx.astype(np.int32), edy.astype(np.int32)
 
 
def detect_pent(image):
    """
 
    :param image: 要预测的图片
    :return: 校准后的预测方框
    """
    num_thresh = 0.7
    scale_factor = 0.709
    P_thresh = 0.5
    model = Pnet()
    
    model.load_weights("./Weights/pnet_wight/pnet_30.ckpt")
 
 
 
    net_size = 12
    current_scale = float(net_size) / min_face_size
 
    im_resized = processed_img(image,current_scale)
    # print("im_resized",im_resized.shape)
    current_h,current_w,_ = im_resized.shape
 
    # im_resize = im_resized.reshape(1, *im_resized.shape)
 
    all_boxes = list()
 
    while min(current_h,current_w) > net_size:
        #因为Pnet要求的数据是[b,w,h,3] 所以在[w,h,3] 0维添加一列
        img_resized = tf.expand_dims(im_resized, axis=0)
        img_resized = tf.cast(img_resized, tf.float32)
 
        cls_prob, bbox_pred, _ = model.predict(img_resized)
 
        cls_prob = cls_prob[0]
        bbox_pred = bbox_pred[0]
 
        bboxes = generate_bounding_box(cls_prob[:,:,1],bbox_pred,current_scale,0.6)
        # print("bboxes",bboxes)
        current_scale *= scale_factor
 
        im_resized = processed_img(image,current_scale)
        current_h, current_w, _ = im_resized.shape
 
 
 
 
        if bboxes.size == 0:
            continue
 
 
        keep = nms(bboxes[:, :5], 0.5)
        bboxes = bboxes[keep]
        all_boxes.append(bboxes)
 
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    keep = nms(all_boxes[:, :5], 0.7)
    all_boxes = all_boxes[keep]
 
    boxes = np.copy(all_boxes[:, :5])
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    x1Arr = all_boxes[:, 0] + all_boxes[:, 5] * bbw
    y1Arr = all_boxes[:, 1] + all_boxes[:, 6] * bbh
    x2Arr = all_boxes[:, 2] + all_boxes[:, 7] * bbw
    y2Arr = all_boxes[:, 3] + all_boxes[:, 8] * bbh
    scoreArr = all_boxes[:, 4]
 
 
    boxes_c = np.concatenate([x1Arr.reshape(-1, 1),
                              y1Arr.reshape(-1, 1),
                              x2Arr.reshape(-1, 1),
                              y2Arr.reshape(-1, 1),
                              scoreArr.reshape(-1, 1)],
                             axis=-1)
    return boxes,boxes_c
 
def detect_Rnet(img,dets):
    '''通过rent选择box
    参数：
      im：输入图像
      dets:pnet选择的box，是相对原图的绝对坐标
    返回值：
      box绝对坐标
    '''
 
    model = Rnet()
    model.load_weights("./Weights/Rnet_wight/rnet_30.ckpt")
    print("successful load")
    h,w,_ = img.shape
    #将pnet的box变成包含他的正方形，可以避免信息损失
 
    dets = convert_to_square(dets)
    # print("dets",dets)
    dets[:,0:4] = np.round(dets[:,0:4])
 
    [dy, edy, dx, edx, y, ey, x, ex, dw, dh] = pad(dets,w,h)
    # print("dy",dw)
    delete_size = np.ones_like(dw) * 20
    ones = np.ones_like(dw)
    zeros = np.zeros_like(dw)
    num_boxes = np.sum(np.where((np.minimum(dw, dh) >= delete_size), ones, zeros))
    cropped_imgs = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
 
    for i in range(num_boxes):
 
 
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if dh[i] < 20 or dw[i] < 20:
            continue
        tmp = np.zeros((dh[i],dw[i],3),dtype=np.uint8)
        tmp[dy[i]:edy[i]+1,dx[i]:edx[i]+1,:] = img[y[i]:ey[i]+1,x[i]:ex[i]+1,:]
 
        cropped_imgs[i,:,:,:] = (cv2.resize(tmp,(24,24)) - 127.5) / 128
 
    cls_scores, reg,_ = model.predict(cropped_imgs)
 
 
 
    cls_scores = cls_scores[:, 1]
 
    keep_inds = np.where(cls_scores > 0.7)[0]
 
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        pass
    else:
        return None, None
        pass
 
    keep = nms(boxes, 0.6)
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes, boxes_c
 
def detect_Onet(img,dets):
 
    """
     将onet的选框继续筛选基本和rnet差不多但多返回了landmark
    :param img:
    :param dets: rnet_
    :return:
    """
    model = Onet()
    model.load_weights("./Weights/Onet_wight/onet_30.ckpt")
    h,w,_ = img.shape
    dets = convert_to_square(dets)
    dets[:,0:4] = np.round(dets[:,0:4])
    [dy, edy, dx, edx, y, ey, x, ex, dw, dh] = pad(dets,w,h)
 
    n_boxes = dets.shape[0]
 
    cropped_imgs = np.zeros((n_boxes,48,48,3),dtype=np.float32)
    for i in range(n_boxes):
 
        tmp = np.zeros((dh[i], dw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        cropped_imgs[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128
 
 
    #cls_scores,reg,linm = model.predict(cropped_imgs)
    #my code
    cls_scores,reg,landmark = model.predict(cropped_imgs)
 
    cls_scores = cls_scores[:,1]
 
    keep_inds = np.where(cls_scores > 0.7)[0]
 
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:,4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None,None,None
 
 
    # h = boxes[:,3] - boxes[:,1] + 1
    # w = boxes[:.2] - boxes[:,0] + 1
    
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
 
    boxes_c = calibrate_box(boxes,reg)
    boxes = boxes[nms(boxes,0.6)]
    keep = nms(boxes_c,0.6)
 
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
 
    return boxes,boxes_c,landmark
