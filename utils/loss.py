# Loss functions

import time
import numpy as np

import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.pose_utils import corner_confidence


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class PoseLoss:
    # Compute losses
    def __init__(self, model, num_keypoints=9, pretrain_num_epochs = 20):
        super(PoseLoss, self).__init__()

        self.device = next(model.parameters()).device  # get model device

        h = model.hyp  # hyperparameters
        self.hyp = h

        self.num_keypoints = num_keypoints
        self.box_loss = nn.L1Loss()


        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        self.obj_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))
        self.cls_loss =  nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=self.device))

        self.pretrain_num_epochs = pretrain_num_epochs
        self.balance = [4.0, 1.0, 0.3, 0.1, 0.03]  # P3-P7
 
        pose = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Pose() module
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(pose.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(pose, k))


    def __call__(self, p, targets, epoch=None):  # predictions, targets, model
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for layer_id, pi in enumerate(p):  # layer index, layer predictions
            
            b, a, gj, gi = indices[layer_id]  # image, anchor, gridy, gridx
            # print(self.device)
            tobj = torch.zeros_like(pi[..., 0]).type(torch.FloatTensor).to(self.device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5 # centroid
                pwh =  compute_new_width_height(ps[:, 2:2*self.num_keypoints])  * anchors[layer_id]
                pbox = torch.cat((pxy, pwh), 1)
                p3dbox = torch.cat((pxy, ps[:, 2:2*self.num_keypoints]), 1)
                # print(f"{tbox[layer_id][:2*self.num_keypoints].shape}")
                # lbox += self.box_loss(p3dbox, tbox[layer_id][:, :2*self.num_keypoints])
                
                # Objectness
                confidence = corner_confidence(tbox[layer_id][:, :2*self.num_keypoints], p3dbox, im_grid_width=p[layer_id].shape[3], im_grid_height=p[layer_id].shape[2], th = 0.25, device = self.device).clamp(min=0) # .detach() .type(tobj.dtype)
                iou = bbox_iou(pbox, tbox[layer_id][:, np.r_[:2, 18:20]], CIoU=True).squeeze()  # iou(prediction, target)

                tobj[b, a, gj, gi] = confidence #  (confidence + iou)/2 # target confidence

                p_edge_keypoints = ps[:, 2:2*self.num_keypoints] * 2. - 0.5
                # print(p_edge_keypoints.shape)
                # print(pxy.shape)
                p_keypoints = torch.cat((pxy, p_edge_keypoints), 1) 
                t_keypoints = tbox[layer_id][:, :self.num_keypoints*2]
                # print(p_keypoints.shape)
                # print(t_keypoints.shape)
                l2_dist = self.box_loss(p_keypoints[:, :2*self.num_keypoints], t_keypoints)
                # l2_dist = (p_keypoints[:, :2*self.num_keypoints][:, 0::2]-t_keypoints[:, 0::2])**2 + (p_keypoints[:, :2*self.num_keypoints][:, 1::2]-t_keypoints[:, 1::2])**2
                # print(f"{l2_dist=}")
                scales = torch.prod(tbox[layer_id][:,-2:], dim=1, keepdim=True)
                # print(scales)
                # print(f"loss: {((1 - torch.exp(-l2_dist/(scales+1e-9)))).mean()}")
                lbox += ((1 - torch.exp(-l2_dist/(scales+1e-9)))).mean()
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 2*self.num_keypoints+1:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[layer_id]] = self.cp
                    lcls += self.cls_loss(ps[:, 2*self.num_keypoints+1:], t) 

            obji = self.obj_loss(pi[..., 2*self.num_keypoints], tobj)
            lobj += obji * self.balance[layer_id] 
            
        lobj *= self.hyp['obj']   
        lbox *= self.hyp['box']
        lcls *= self.hyp['cls']

        if epoch is not None and epoch > self.pretrain_num_epochs: 
            loss  = lbox + lobj + lcls
        else:
            loss  = lbox + lcls
        bs = tobj.shape[0]  # batch size

        return loss*bs, torch.cat((lobj, lbox, lcls)).detach()


    
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(23, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) # anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            shape = p[i].shape
            gain[2:22] = torch.tensor(p[i].shape)[10*[3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 20:22] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gxy_coords = t[:, 4:2*self.num_keypoints+2]  # other box coords
            gwh = t[:, 2*self.num_keypoints+2: 2*self.num_keypoints+4]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            for key_idx in range(self.num_keypoints-1):
                gxy_coords[:, key_idx*2: key_idx*2+2] = gxy_coords[:, key_idx*2: key_idx*2+2]-gij

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gxy_coords, gwh), 1))  # box

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


def compute_new_width_height(coordinates):
    # print((torch.amax(coordinates[:, ::2], dim=1)-torch.amin(coordinates[:, ::2], dim=1)).shape )
    return torch.stack([torch.amax(coordinates[:, ::2], dim=1)-torch.amin(coordinates[:, ::2], dim=1) , torch.amax(coordinates[:, 1::2], dim=1)-torch.amin(coordinates[:, 1::2], dim=1)], dim=1)

