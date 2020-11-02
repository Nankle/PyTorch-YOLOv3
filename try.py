
'''
import glob

path = 'data/custom/train.txt'
a = sorted(glob.glob("%s/*.*" % path))
print(a)
'''

import torch
import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint #多边形

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    # b, tt = wh2.long().t()
    # print('***********bb', wh2.long().t())
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    print(f'Torch.minw:{torch.min(w1, w2)}')
    print(f'Torch.minh:{torch.min(h1, h2)}')
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    print(f'inter_area:{inter_area}')
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def calculate_Iou():

    line1=[[2,0,2,2,0,0,0,2], [2,1,2,4,0,0,0,6], [2,0,2,2,0,0,0,2]]  #四边形四个点坐标的一维数组表示，[x,y,x,y....]
    a=np.array(line1).reshape(-1, 2)  #四边形二维坐标表示
    poly1 = Polygon(a).convex_hull #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    print(Polygon(a).convex_hull) #可以打印看看是不是这样子

    line2=[[1,1,4,1,4,4,1,4], [1,1,4,1,4,4,1,4], [1,1,4,1,4,4,1,4]]
    b=np.array(line2).reshape(-1, 2)
    poly2 = Polygon(b).convex_hull
    print(Polygon(b).convex_hull)
    
    union_poly = np.concatenate((a,b))  #合并两个box坐标，变为8*2
    #print(union_poly)
    print("Lemon:", MultiPoint(union_poly).convex_hull)   #包含两四边形最小的多边形点
    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  #相交面积
            print(inter_area)
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            print(union_area)
            if union_area == 0:
                iou= 0
            #iou = float(inter_area) / (union_area-inter_area) #错了
            iou=float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积 
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    print(a)
    
    print(iou)

if __name__ == "__main__":
    # anchors = torch.Tensor([(100,125),(34,45),(43,78)])
    # targets = torch.Tensor([[77.1,78.2],[87.1,88.2],[97,98],[17,28],[47,58]])
    # print(targets.shape)
    # ious = torch.stack([bbox_wh_iou(anchor, targets) for anchor in anchors])
    # best_ious, best_n = ious.max(0)  # 0-dim 按列找最大值
    # # print(ious)
    # print(best_ious, best_n)

    # obj_mask = torch.Tensor(4, 3, 4, 4).fill_(0)
    # print(obj_mask.shape)
    # obj_mask[1 , 1, 0, 0] = 1

    # print('*****************************')
    # print(obj_mask[1,1,0])
    # tensor([[0.4805, 0.6125, 0.7605, 0.0381, 0.2181],
    #     [0.2547, 0.1998, 0.1610, 0.3111, 0.5613],
    #     [0.5584, 0.4381, 0.3528, 0.1419, 0.6955]])
    # tensor([0.5584, 0.6125, 0.7605, 0.3111, 0.6955]) 
    # tensor([2, 0, 0, 1, 2])

    # a = torch.Tensor([[0.4805, 0.6125, 0.7605, 0.0381, 0.2181],\
    #     [0.2547, 0.1998, 0.1610, 0.3111, 0.5613],\
    #         [0.5584, 0.4381, 0.3528, 0.1419, 0.6955]])
    # b = [[0.4805, 0.6125, 0.7605, 0.0381, 0.2181],\
    #     [0.2547, 0.1998, 0.1610, 0.3111, 0.5613],\
    #         [0.5584, 0.4381, 0.3528, 0.1419, 0.6955]] 
    # b[:,0] = 1  
    # b = torch.cat(b, 0)
    # print(b)
    # pridiction = a.view(1,-1,3,1)
    # print(pridiction.shape)
    # print(pridiction)
    # b = torch.Tensor([3,3,3]).view(1,1,3,1)
    # print(b)
    # print(pridiction.data*b)

    # print(torch.zeros(10, dtype=torch.int))
    # calculate_Iou()

    a = torch.Tensor([10])
    b = torch.Tensor([12])
    c = torch.Tensor([13])
    print(a,b,c)
    print(a.numpy(), b.numpy(), c.numpy())
    print(np.concatenate((a,b,c)))
    