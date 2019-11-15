import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def build_targets(pred_corners, target, num_keypoints, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask    = torch.zeros(nB, nA, nH, nW)
    txs = list()
    tys = list()
    for i in range(num_keypoints):
        txs.append(torch.zeros(nB, nA, nH, nW))
        tys.append(torch.zeros(nB, nA, nH, nW)) 
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls  = torch.zeros(nB, nA, nH, nW) 

    num_labels = 2 * num_keypoints + 3 # +2 for width, height and +1 for class within label files
    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_corners = pred_corners[b*nAnchors:(b+1)*nAnchors].t()
        cur_confs = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*num_labels+1] == 0:
                break
            g = list()
            for i in range(num_keypoints):
                g.append(target[b][t*num_labels+2*i+1])
                g.append(target[b][t*num_labels+2*i+2])

            cur_gt_corners = torch.FloatTensor(g).repeat(nAnchors,1).t() # 16 x nAnchors
            cur_confs  = torch.max(cur_confs, corner_confidences(cur_pred_corners, cur_gt_corners)).view_as(conf_mask[b]) # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
        conf_mask[b][cur_confs>sil_thresh] = 0


    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*num_labels+1] == 0:
                break
            # Get gt box for the current label
            nGT = nGT + 1
            gx = list()
            gy = list()
            gt_box = list()
            for i in range(num_keypoints):
                gt_box.extend([target[b][t*num_labels+2*i+1], target[b][t*num_labels+2*i+2]])
                gx.append(target[b][t*num_labels+2*i+1] * nW)
                gy.append(target[b][t*num_labels+2*i+2] * nH)
                if i == 0:
                    gi0  = int(gx[i])
                    gj0  = int(gy[i])
            # Update masks
            best_n = 0 # 1 anchor box
            pred_box = pred_corners[b*nAnchors+best_n*nPixels+gj0*nW+gi0]
            conf = corner_confidence(gt_box, pred_box) 
            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0]   = 1
            conf_mask[b][best_n][gj0][gi0]  = object_scale
            # Update targets
            for i in range(num_keypoints):
                txs[i][b][best_n][gj0][gi0] = gx[i]- gi0
                tys[i][b][best_n][gj0][gi0] = gy[i]- gj0   
            tconf[b][best_n][gj0][gi0]      = conf
            tcls[b][best_n][gj0][gi0]       = target[b][t*num_labels]
            # Update recall during training
            if conf > 0.5: 
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls
           
class RegionLoss(nn.Module):
    def __init__(self, num_keypoints=9, num_classes=1, anchors=[], num_anchors=1, pretrain_num_epochs=15):
        # Define the loss layer
        super(RegionLoss, self).__init__()
        self.num_classes         = num_classes
        self.num_anchors         = num_anchors # for single object pose estimation, there is only 1 trivial predictor (anchor)
        self.num_keypoints       = num_keypoints
        self.coord_scale         = 1
        self.noobject_scale      = 1
        self.object_scale        = 5
        self.class_scale         = 1
        self.thresh              = 0.6
        self.seen                = 0
        self.pretrain_num_epochs = pretrain_num_epochs

    def forward(self, output, target, epoch):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        num_keypoints = self.num_keypoints

        # Activation
        output = output.view(nB, nA, (num_keypoints*2+1+nC), nH, nW)
        x = list()
        y = list()
        x.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW)))
        y.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW)))
        for i in range(1,num_keypoints):
            x.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 0]))).view(nB, nA, nH, nW))
            y.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 1]))).view(nB, nA, nH, nW))
        conf   = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([2 * num_keypoints]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(2*num_keypoints+1,2*num_keypoints+1+nC-1,nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(2*num_keypoints, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        for i in range(num_keypoints):
            pred_corners[2 * i + 0]  = (x[i].data.view_as(grid_x) + grid_x) / nW
            pred_corners[2 * i + 1]  = (y[i].data.view_as(grid_y) + grid_y) / nH
        gpu_matrix = pred_corners.transpose(0,1).contiguous().view(-1,2*num_keypoints)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls = \
                       build_targets(pred_corners, target.data, num_keypoints, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])
        for i in range(num_keypoints):
            txs[i] = Variable(txs[i].cuda())
            tys[i] = Variable(tys[i].cuda())
        tconf      = Variable(tconf.cuda())
        tcls       = Variable(tcls[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  
        t3 = time.time()

        # Create loss
        loss_xs   = list()
        loss_ys   = list()
        for i in range(num_keypoints):
            loss_xs.append(self.coord_scale * nn.MSELoss(size_average=False)(x[i]*coord_mask, txs[i]*coord_mask)/2.0)
            loss_ys.append(self.coord_scale * nn.MSELoss(size_average=False)(y[i]*coord_mask, tys[i]*coord_mask)/2.0)
        loss_conf  = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_x    = np.sum(loss_xs)
        loss_y    = np.sum(loss_ys)

        if epoch > self.pretrain_num_epochs:
            loss  = loss_x + loss_y + loss_conf # in single object pose estimation, there is no classification loss
        else:
            # pretrain initially without confidence loss
            # once the coordinate predictions get better, start training for confidence as well
            loss  = loss_x + loss_y 

        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_conf.data[0], loss.data[0]))
        
        return loss



class DistiledRegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(DistiledRegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target, distiled_target):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # Activation
        output = output.view(nB, nA, (19+nC), nH, nW)
        x0     = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y0     = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        x1     = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        y1     = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        x2     = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
        y2     = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
        x3     = output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW)
        y3     = output.index_select(2, Variable(torch.cuda.LongTensor([7]))).view(nB, nA, nH, nW)
        x4     = output.index_select(2, Variable(torch.cuda.LongTensor([8]))).view(nB, nA, nH, nW)
        y4     = output.index_select(2, Variable(torch.cuda.LongTensor([9]))).view(nB, nA, nH, nW)
        x5     = output.index_select(2, Variable(torch.cuda.LongTensor([10]))).view(nB, nA, nH, nW)
        y5     = output.index_select(2, Variable(torch.cuda.LongTensor([11]))).view(nB, nA, nH, nW)
        x6     = output.index_select(2, Variable(torch.cuda.LongTensor([12]))).view(nB, nA, nH, nW)
        y6     = output.index_select(2, Variable(torch.cuda.LongTensor([13]))).view(nB, nA, nH, nW)
        x7     = output.index_select(2, Variable(torch.cuda.LongTensor([14]))).view(nB, nA, nH, nW)
        y7     = output.index_select(2, Variable(torch.cuda.LongTensor([15]))).view(nB, nA, nH, nW)
        x8     = output.index_select(2, Variable(torch.cuda.LongTensor([16]))).view(nB, nA, nH, nW)
        y8     = output.index_select(2, Variable(torch.cuda.LongTensor([17]))).view(nB, nA, nH, nW)
        conf   = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([18]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(19,19+nC-1,nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(18, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        pred_corners[0]  = (x0.data.view_as(grid_x) + grid_x) / nW
        pred_corners[1]  = (y0.data.view_as(grid_y) + grid_y) / nH
        pred_corners[2]  = (x1.data.view_as(grid_x) + grid_x) / nW
        pred_corners[3]  = (y1.data.view_as(grid_y) + grid_y) / nH
        pred_corners[4]  = (x2.data.view_as(grid_x) + grid_x) / nW
        pred_corners[5]  = (y2.data.view_as(grid_y) + grid_y) / nH
        pred_corners[6]  = (x3.data.view_as(grid_x) + grid_x) / nW
        pred_corners[7]  = (y3.data.view_as(grid_y) + grid_y) / nH
        pred_corners[8]  = (x4.data.view_as(grid_x) + grid_x) / nW
        pred_corners[9]  = (y4.data.view_as(grid_y) + grid_y) / nH
        pred_corners[10] = (x5.data.view_as(grid_x) + grid_x) / nW
        pred_corners[11] = (y5.data.view_as(grid_y) + grid_y) / nH
        pred_corners[12] = (x6.data.view_as(grid_x) + grid_x) / nW
        pred_corners[13] = (y6.data.view_as(grid_y) + grid_y) / nH
        pred_corners[14] = (x7.data.view_as(grid_x) + grid_x) / nW
        pred_corners[15] = (y7.data.view_as(grid_y) + grid_y) / nH
        pred_corners[16] = (x8.data.view_as(grid_x) + grid_x) / nW
        pred_corners[17] = (y8.data.view_as(grid_y) + grid_y) / nH
        gpu_matrix = pred_corners.transpose(0,1).contiguous().view(-1,18)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls = \
                       build_targets(pred_corners, target.data, self.anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().item())
        distiled_target = distiled_target.view(nB, nA, (19+nC), nH, nW)
        tx0     = torch.sigmoid(distiled_target.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        ty0     = torch.sigmoid(distiled_target.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        tx1     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        ty1     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        tx2     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
        ty2     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
        tx3     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW)
        ty3     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([7]))).view(nB, nA, nH, nW)
        tx4     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([8]))).view(nB, nA, nH, nW)
        ty4     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([9]))).view(nB, nA, nH, nW)
        tx5     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([10]))).view(nB, nA, nH, nW)
        ty5     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([11]))).view(nB, nA, nH, nW)
        tx6     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([12]))).view(nB, nA, nH, nW)
        ty6     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([13]))).view(nB, nA, nH, nW)
        tx7     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([14]))).view(nB, nA, nH, nW)
        ty7     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([15]))).view(nB, nA, nH, nW)
        tx8     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([16]))).view(nB, nA, nH, nW)
        ty8     = distiled_target.index_select(2, Variable(torch.cuda.LongTensor([17]))).view(nB, nA, nH, nW)
        tconf    = torch.sigmoid(distiled_target.index_select(2, Variable(torch.cuda.LongTensor([18]))).view(nB, nA, nH, nW))
        tcls       = Variable(tcls[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)

        t3 = time.time()

        # Create loss
        loss_x0    = self.coord_scale * nn.MSELoss(size_average=False)(x0*coord_mask, tx0*coord_mask)/2.0
        loss_y0    = self.coord_scale * nn.MSELoss(size_average=False)(y0*coord_mask, ty0*coord_mask)/2.0
        loss_x1    = self.coord_scale * nn.MSELoss(size_average=False)(x1*coord_mask, tx1*coord_mask)/2.0
        loss_y1    = self.coord_scale * nn.MSELoss(size_average=False)(y1*coord_mask, ty1*coord_mask)/2.0
        loss_x2    = self.coord_scale * nn.MSELoss(size_average=False)(x2*coord_mask, tx2*coord_mask)/2.0
        loss_y2    = self.coord_scale * nn.MSELoss(size_average=False)(y2*coord_mask, ty2*coord_mask)/2.0
        loss_x3    = self.coord_scale * nn.MSELoss(size_average=False)(x3*coord_mask, tx3*coord_mask)/2.0
        loss_y3    = self.coord_scale * nn.MSELoss(size_average=False)(y3*coord_mask, ty3*coord_mask)/2.0
        loss_x4    = self.coord_scale * nn.MSELoss(size_average=False)(x4*coord_mask, tx4*coord_mask)/2.0
        loss_y4    = self.coord_scale * nn.MSELoss(size_average=False)(y4*coord_mask, ty4*coord_mask)/2.0
        loss_x5    = self.coord_scale * nn.MSELoss(size_average=False)(x5*coord_mask, tx5*coord_mask)/2.0
        loss_y5    = self.coord_scale * nn.MSELoss(size_average=False)(y5*coord_mask, ty5*coord_mask)/2.0
        loss_x6    = self.coord_scale * nn.MSELoss(size_average=False)(x6*coord_mask, tx6*coord_mask)/2.0
        loss_y6    = self.coord_scale * nn.MSELoss(size_average=False)(y6*coord_mask, ty6*coord_mask)/2.0
        loss_x7    = self.coord_scale * nn.MSELoss(size_average=False)(x7*coord_mask, tx7*coord_mask)/2.0
        loss_y7    = self.coord_scale * nn.MSELoss(size_average=False)(y7*coord_mask, ty7*coord_mask)/2.0
        loss_x8    = self.coord_scale * nn.MSELoss(size_average=False)(x8*coord_mask, tx8*coord_mask)/2.0
        loss_y8    = self.coord_scale * nn.MSELoss(size_average=False)(y8*coord_mask, ty8*coord_mask)/2.0
        loss_conf  = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        # loss_cls   = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss_cls = 0
        loss_x     = loss_x0 + loss_x1 + loss_x2 + loss_x3 + loss_x4 + loss_x5 + loss_x6 + loss_x7 + loss_x8 
        loss_y     = loss_y0 + loss_y1 + loss_y2 + loss_y3 + loss_y4 + loss_y5 + loss_y6 + loss_y7 + loss_y8 
        if False:
            loss   = loss_x + loss_y + loss_conf + loss_cls
        else:
            loss   = loss_x + loss_y + loss_conf
        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        if False:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        else:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss.item()))
        
        return loss