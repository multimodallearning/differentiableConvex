import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import numpy as np
import torchvision
import imageio
import os


def soft_dice(fixed,moving):
    B,C,H,W = fixed.shape
    dice = (2. * fixed*moving).reshape(B,-1,H*W).mean(2) / (1e-8 + fixed.reshape(B,-1,H*W).mean(2) + moving.reshape(B,-1,H*W).mean(2))
    return dice


def adapt_resnet():
    resnet = torchvision.models.resnet18(pretrained=False)
    #Total params: 11,689,512
    #Total mult-adds (G): 2.22
    resnet.fc = nn.Identity()
    resnet.conv1 = nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)

    resnet.layer3 = nn.Sequential(nn.Conv2d(128,128,3,stride=1,padding=2,dilation=2,bias=False),\
                                  nn.BatchNorm2d(128),nn.ReLU(inplace=True),\
                                  nn.Conv2d(128,128,3,stride=1,padding=1,bias=False),\
                                  nn.BatchNorm2d(128),nn.ReLU(inplace=True),\
                                  nn.Conv2d(128,64,3,stride=1,padding=1,bias=True),nn.Tanh())

    resnet.layer4 = nn.Identity()
    resnet.maxpool = nn.Identity()
    resnet.avgpool = nn.Identity()#nn.Upsample(scale_factor=2,mode='bilinear')
    return resnet

#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp):
    torch.cuda.synchronize()
    _,C_mind,H_grid,W_grid = mind_fix.shape
    t0 = time.time()
    #with torch.no_grad():
    mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw)),disp_hw*2+1)
    mind_unfold = mind_unfold.view(C_mind,-1,H_grid,W_grid)
    mind_sum = (mind_fix.transpose(1,0)-mind_unfold).abs().sum(0,keepdim=True)
    ssd = F.avg_pool2d(mind_sum,3,stride=1,padding=1).squeeze(1)
    #print(mind_sum.shape)
    return ssd




##DIFFERENTIABLE VERSION OF COUPLED-CONVEX
#solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex(ssd,disp_mesh_t,grid_sp,alpha=20,coeffs=torch.tensor([0.003,0.03,0.3])):
    _,_,H_grid,W_grid = ssd.shape
    disp_soft = F.avg_pool2d((disp_mesh_t.view(2,-1,1,1)*torch.softmax(-alpha*ssd,1)).sum(1).unsqueeze(0),3,padding=1,stride=1)

    for j in range(len(coeffs)):
        coupling = coeffs[j]*(disp_mesh_t-disp_soft.view(2,1,-1)).pow(2).sum(0).view(-1,H_grid,W_grid)
        
        disp_soft = F.avg_pool2d((disp_mesh_t.view(2,-1,1,1)*torch.softmax(-alpha*(ssd+coupling),1)).sum(1).unsqueeze(0),3,padding=1,stride=1)


    return disp_soft



def jacobian_det2d(rand_field):
    B,_,H,W = rand_field.size()
    rep_x = nn.ReplicationPad2d((1,1,0,0))
    rep_y = nn.ReplicationPad2d((0,0,1,1))


    kernel_y = nn.Conv2d(2,2,(3,1),bias=False,groups=2)
    kernel_y.cuda()
    kernel_y.weight.data[:,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1).to(rand_field.device)
    kernel_x = nn.Conv2d(2,2,(1,3),bias=False,groups=2)
    kernel_x.cuda()
    kernel_x.weight.data[:,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1).to(rand_field.device)

    rand_field_vox = rand_field.flip(1)*(torch.Tensor([H-1,W-1]).to(rand_field.device).view(1,2,1,1)-1)/2
    grad_y = kernel_y(rep_y(rand_field_vox))
    grad_x = kernel_x(rep_x(rand_field_vox))

    jacobian = torch.stack((grad_y,grad_x),1)+torch.eye(2,2).to(rand_field.device).view(1,2,2,1,1)#.to(dense_flow.device)
    jac_det = jacobian[:,0,0,:,:]*jacobian[:,1,1,:,:] - jacobian[:,1,0,:,:]*jacobian[:,0,1,:,:]
    return jac_det

def MIND2D_64(image,layout,grid): #layout should be of size 2x192x1x2
    #batch and channels should be equal to 1
    B,C,H,W = image.size()
    
    
    #smaller fixed length offsets for 64 MIND-SSC like features
    brief_layout3 = layout[0:1,0:,:,:]*0.15
    brief_layout4 = layout[1:2,0:,:,:]*0.15
    brief_layout4[:,:32,:,:] = 0
    fixed_length = 0.15
    brief_length = torch.sqrt(torch.sum((brief_layout3-brief_layout4)**2,3,keepdim=True))
    brief_layout3 /= (brief_length/fixed_length)
    brief_layout4 /= (brief_length/fixed_length)
    
    img_patch = F.unfold(image,5,padding=2).view(1,25,H,W)
    brief_patch = torch.sum((F.grid_sample(img_patch,brief_layout3+grid.view(1,1,-1,2),align_corners=True)-F.grid_sample(img_patch,brief_layout4+grid.view(1,1,-1,2),align_corners=True))**2,1)
    brief_patch -= brief_patch.min(1)[0]
    brief_patch /= torch.clamp_min(brief_patch.std(1),1e-5)
    brief_patch = torch.exp(-brief_patch).view(1,-1,grid.size(1),grid.size(2))
    
    return brief_patch#torch.cat((brief_patch,brief_context),1)


def create_dataset():
    imgs_all = []
    labels_all = []
    labels2_all = []

    for ix in range(1,6):
        for j in range(1,3):
            img = torch.from_numpy(imageio.imread('CLUST_png/ETH-0'+str(ix)+'-'+str(j)+'/Data/00001.png')).float()
            annotation = torch.from_numpy(np.loadtxt('CLUST_png/ETH-0'+str(ix)+'-'+str(j)+'/Annotation/ETH-0'+str(ix)+'-'+str(j)+'_1.txt')).float()
            annotation_ = annotation[:,1:]
            file2 = 'CLUST_png/ETH-0'+str(ix)+'-'+str(j)+'/Annotation/ETH-0'+str(ix)+'-'+str(j)+'_2.txt'
            if(os.path.exists(file2)):
                annotation2 = torch.from_numpy(np.loadtxt(file2)).float()
                annotation2_ = annotation2[:,1:]



            imgs = torch.zeros(len(annotation), 1, 160+32, 192+32).cuda()
            labels = torch.zeros(len(annotation), 4, 160+32, 192+32).cuda()
            labels12 = torch.zeros(len(annotation), 8, 160+32, 192+32)#.cuda()
            mesh = torch.stack(torch.meshgrid(torch.arange(img.shape[0]),torch.arange(img.shape[1]))).cuda()
            for i in range(len(annotation)):
                idx = int(annotation[i,0])
                label = F.interpolate((mesh-annotation_[i].cuda().view(2,1,1).flip(0)).pow(2).sum(0).sqrt().unsqueeze(0).unsqueeze(0),size=(160+32, 192+32),mode='bilinear').cuda().squeeze(1)
                label1 = (label<35).long()+(label<5).long()+(label<20).long()
                labels12[i,:4] = F.one_hot(label1,4).permute(0,3,1,2).cpu()
                if(os.path.exists(file2)):
                    label = F.interpolate((mesh-annotation2_[i].cuda().view(2,1,1).flip(0)).pow(2).sum(0).sqrt().unsqueeze(0).unsqueeze(0),size=(160+32, 192+32),mode='bilinear').cuda().squeeze(1)
                    label2 = (label<35).long()+(label<5).long()+(label<20).long()
                    label1 = torch.maximum(label1,label2)
                    labels12[i,4:] = F.one_hot(label2,4).permute(0,3,1,2).cpu()


                labels[i] = F.one_hot(label1,4).permute(0,3,1,2).cpu()
                img = torch.from_numpy(imageio.imread('CLUST_png/ETH-0'+str(ix)+'-'+str(j)+'/Data/'+str(idx).zfill(5)+'.png')).float().cuda()/150
                imgs[i] = F.interpolate(img.unsqueeze(0).unsqueeze(0),size=(160+32, 192+32),mode='bilinear')
            imgs /= imgs.reshape(-1,192*224).std(1).mean()

            imgs_all.append(imgs)
            labels_all.append(labels)
            labels2_all.append(labels12)
            
    return imgs_all,labels_all,labels2_all
