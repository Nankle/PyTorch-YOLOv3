
'''
import glob

path = 'data/custom/train.txt'
a = sorted(glob.glob("%s/*.*" % path))
print(a)
'''

import torch

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    # b, tt = wh2.long().t()
    print('***********bb', wh2.long().t())
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    print(f'Torch.min:{torch.min(w1, w2)}')
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


if __name__ == "__main__":
    # anchors = torch.Tensor([(100,125),(34,45),(43,78)])
    # targets = torch.Tensor([[77.1,78.2],[87.1,88.2],[97,98],[17,28],[47,58]])
    # ious = torch.stack([bbox_wh_iou(anchor, targets) for anchor in anchors])
    # best_ious, best_n = ious.max(0)  # 0-dim 按列找最大值
    # print(ious)
    # print(best_ious, best_n)

    # tensor([[0.4805, 0.6125, 0.7605, 0.0381, 0.2181],
    #     [0.2547, 0.1998, 0.1610, 0.3111, 0.5613],
    #     [0.5584, 0.4381, 0.3528, 0.1419, 0.6955]])
    # tensor([0.5584, 0.6125, 0.7605, 0.3111, 0.6955]) 
    # tensor([2, 0, 0, 1, 2])

    a = torch.Tensor([[0.4805, 0.6125, 0.7605, 0.0381, 0.2181],\
        [0.2547, 0.1998, 0.1610, 0.3111, 0.5613],\
            [0.5584, 0.4381, 0.3528, 0.1419, 0.6955]])
    pridiction = a.view(1,-1,3,1)
    print(pridiction.shape)
    print(pridiction)
    b = torch.Tensor([3,3,3]).view(1,1,3,1)
    print(b)
    print(pridiction.data*b)

    