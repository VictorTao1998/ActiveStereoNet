import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RHLoss(nn.Module):

    def __init__(self, max_disp):

        super(RHLoss, self).__init__()
        self.max_disp = max_disp
        self.crit = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, output, target):

        #mask = (target < self.max_disp).float()
        #output *= mask
        #target *= mask
        mask = target < self.max_disp
        mask.detach_()
        
        loss = self.crit(output[mask], target[mask])
        
        return loss



class XTLoss(nn.Module):
    '''
    Args:
        left_img right_img: N * C * H * W,
        dispmap : N * H * W
    '''
    def __init__(self, max_disp):
        super(XTLoss, self).__init__()
        self.max_disp = max_disp
        self.theta = torch.Tensor(
            [[1, 0, 0],  # 控制左右，-右，+左
            [0, 1, 0]]    # 控制上下，-下，+上
        )
        self.inplanes = 1
        self.outplanes = 1
        


    
    def forward(self, left_img, right_img, dispmap):

        n, c, h, w = left_img.shape
        
        #pdb.set_trace()
        theta = self.theta.repeat(left_img.size()[0], 1, 1)
        
        
        grid = F.affine_grid(theta, left_img.size())
        grid = grid.cuda()
        
        dispmap_norm = dispmap * 2 / w
        dispmap_norm = dispmap_norm.cuda()
        #pdb.set_trace()
        dispmap_norm = dispmap_norm.squeeze(1).unsqueeze(3)
        dispmap_norm = torch.cat((dispmap_norm, torch.zeros(dispmap_norm.size()).cuda()), dim=3)
        
        grid -= dispmap_norm
        
        recon_img = F.grid_sample(right_img, grid)
        
        #pdb.set_trace()
        recon_img_LCN, _, _ = self.LCN(recon_img, 9)
        
        left_img_LCN, _, left_std_local = self.LCN(left_img, 9)
        
        #pdb.set_trace()
        losses = torch.abs(((left_img_LCN - recon_img_LCN) * left_std_local))
        
        #pdb.set_trace()
        losses = self.ASW(left_img, losses, 12, 2)
        
        return losses


    def LCN(self, img, kSize):
        '''
            Args: 
                img : N * C * H * W
                kSize : 9 * 9
        '''

        w = torch.ones((self.outplanes, self.inplanes, kSize, kSize)).cuda() / (kSize * kSize)
        mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2)
        
        mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2)
        std_local = (mean_square_local - mean_local ** 2) * (kSize ** 2) / (kSize ** 2 - 1)
        
        epsilon = 1e-6
        
        return (img - mean_local) / (std_local + epsilon), mean_local, std_local


    def ASW(self, img, Cost, kSize, sigma_omega):
        
        #pdb.set_trace()
        weightGraph = torch.zeros(img.shape, requires_grad=False).cuda()
        CostASW = torch.zeros(Cost.shape, dtype=torch.float, requires_grad=True).cuda()

        pad_len = kSize // 2
        img = F.pad(img, [pad_len] * 4)
        Cost = F.pad(Cost, [pad_len] * 4)
        n, c, h, w = img.shape
        #pdb.set_trace()
        

        
        for i in range(kSize):
            for j in range(kSize):
                tempGraph = torch.abs(img[:, :, pad_len : h - pad_len, pad_len : w - pad_len] - img[:, :, i:i + h - pad_len * 2, j:j + w - pad_len * 2])
                tempGraph = torch.exp(-tempGraph / sigma_omega)
                weightGraph += tempGraph
                CostASW += tempGraph * Cost[:, :, i:i + h - pad_len * 2, j:j + w - pad_len * 2]
    
        CostASW = CostASW / weightGraph

        return CostASW.mean()


        
