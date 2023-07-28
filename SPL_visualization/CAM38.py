import sys
sys.path.append("/media/ders/mazhiming/eps_test/EPS-main/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch.nn.functional as F
from torchvision.io.image import read_image

from torchvision.models import resnet50,vgg16
# from models import resnet38
# import models
# from torchcam.methods import GradCAM
# from torchcam.methods import CAM as get_cam_map
import importlib
import argparse
import torch
from tools.ai.torch_utils import *
from tools.ai.demo_utils import *
from tools.general.Q_util import *
import core.models as fcnmodel
import torch
import torch.nn as nn
import models.resnet38_base
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib.pyplot as plt


set_seed(0)
def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of L2G')
    parser.add_argument("--num_classes", type=int, default=20)
    return parser.parse_args()


def vispcam(images,features,mask):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    cams = (make_cam(features)[:,:-1] * mask)
    obj_cams = cams.max(dim=1)[0]
    image = get_numpy_from_tensor(images)
    cam = get_numpy_from_tensor(obj_cams)[0]

    image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
    h, w, c = image.shape

    cam = (cam * 255).astype(np.uint8)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = colormap(cam)

    image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::]
    image = image.astype(np.float32) / 255.
    return image

def visimg(images):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image = get_numpy_from_tensor(images)
    image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
    h, w, c = image.shape

    # image = image[..., ::]
    image = image.astype(np.float32) 
    return image

def cam_vis(model,input_tensor,name):
    savepth='experiments/CAMtest/cam/'+name
    cam,out1 = model(input_tensor.unsqueeze(0))
    # cam=F.softmax(cam,dim=1)
    mask=torch.zeros_like(out1).squeeze()
    mask[out1.squeeze(0).argmax().item()]=1
    mask=mask.unsqueeze(0)
    img = vispcam(input_tensor,cam,mask.unsqueeze(2).unsqueeze(3))*255
    print("out1="+str(out1.squeeze(0).argmax().item()))
    cv2.imwrite(savepth,img)
    return out1
    
def sp_cam_vis(model,input_tensor,name,out1):
    savepth='experiments/CAMtest/sp_img_cam/'+name
    savepth_spimg='experiments/CAMtest/sp_img/'+name

    cam,out1 = model(input_tensor)

    mask=torch.zeros_like(out1).squeeze()
    mask[out1.squeeze(0).argmax().item()]=1
    # mask[4]=1
    mask=mask.unsqueeze(0)
    img = vispcam(input_tensor.squeeze(0),cam,mask.unsqueeze(2).unsqueeze(3))*255
    
    sp_img=visimg(input_tensor.squeeze(0))
    
    print("out2="+str(out1.squeeze(0).argmax().item()))
    cv2.imwrite(savepth,img)
    cv2.imwrite(savepth_spimg,sp_img)
    return cam,mask

def camp_vis(map,Qs,input_tensor,mask):
    default_size_h,default_size_w=16,16
    c,h,w=input_tensor.size()
    map = F.interpolate(map,size=(int(h/16),int(w/16)),mode='bilinear', align_corners=False)
    # map=F.softmax(map,dim=1)
    # map=torch.where(map>0.9,torch.ones_like(map)*0.99,torch.ones_like(map)*0.01)
    affmats = calc_affmat(Qs)
    renum=1000
    # with torch.no_grad():
    for i in range(renum):
        map = refine_with_affmat(map, affmats)
    
    map = F.interpolate(map,size=(int(h*2),int(w*2)),mode='bilinear', align_corners=False)
    
    savepth='CAMtest/cam+/'+name

    img = vispcam(input_tensor.squeeze(0),map,mask.unsqueeze(2).unsqueeze(3))*255
    cv2.imwrite(savepth,img)
    # return sp_map

args = get_arguments()
#get model(pretrained)
model = getattr(importlib.import_module("models.resnet38"), 'Net')(args)
weights_dict = models.resnet38_base.convert_mxnet_to_torch('./models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
weights_dict = torch.load('experiments/log/voc_dyrenum65_thr0.8_25ep/best_checkpoint.pth')
# weights_dict = torch.load('log/eps_2epoch_/best_checkpoint.pth')
model.load_state_dict(weights_dict, strict=False)

model.cuda().eval()

#get Qmodel
Q_model = fcnmodel.SpixelNet1l_bn().cuda()
Q_model.load_state_dict(torch.load('/media/ders/mazhiming/SP_CAM_code/SPCAM/experiments/models/train_Q_/00.pth'))
Q_model = nn.DataParallel(Q_model)
Q_model.eval()

#get img
name='2008_003429.jpg'
img = read_image("CAMtest/review_img/"+name).cuda()
C,H,W=img.size()
input_tensor = normalize(resize(img, (448, 448)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# input_tensor = input_tensor[:,0:224,0:224]
#get Q
Q = Q_model(input_tensor.unsqueeze(0))
size_sp=16
#get sp_img

# sal=Image.open('experiments/CAMtest/img/2007_000480.png').convert('L')
# import utilsl2g.transforms.functional as Fu
# sal = Fu.to_tensor(sal).unsqueeze(0).cuda()
# sal=F.interpolate(sal,(448,448))
# upf=sal.squeeze(0).detach().cpu()
# pil_upf=to_pil_image(upf,mode='L')#.savefig('experiments/CAMtest/sp_sal/a.png')
# # pil_upf=pil_upf.convert('RGB')
# plt.imshow(pil_upf);plt.savefig('experiments/CAMtest/sp_sal/1.jpg')

# # sal=torch.where(0.5>sal,torch.ones_like(sal)*0.5,sal)
# sal=torch.where(sal>0,torch.ones_like(sal),sal)
# sal=torch.where(sal==0,torch.ones_like(sal)*0.001,sal)
# upf=poolfeat(sal,Q)
# upf=F.interpolate(upf,(448,448),mode='bilinear')
# # upf=upfeat(poolf,Q)
# upf=torch.where(upf<=0.001,torch.zeros_like(upf),upf)
# upf=upf.squeeze(0).detach().cpu()
# pil_upf=to_pil_image(upf,mode='L')#.savefig('experiments/CAMtest/sp_sal/a.png')
# # pil_upf=pil_upf.convert('RGB')
# plt.imshow(pil_upf);plt.savefig('experiments/CAMtest/sp_sal/a.jpg')
# pil_upf.save('experiments/CAMtest/sp_sal/a.jpg')
# cv2.imwrite('experiments/CAMtest/sp_sal/a.png',upf)



input_sp=input_tensor.unsqueeze(0)
# input_sp = poolfeat(input_sp, Q, size_sp, size_sp)
# input_sp = upfeat(input_sp, Q, size_sp, size_sp)
# input_sp = F.interpolate(input_sp,(28,28),mode='bilinear', align_corners=False)
# input_sp = upfeat(input_sp, Q, size_sp, size_sp)
# input_sp = poolfeat(input_sp, Q, size_sp, size_sp)

# input_sp = F.interpolate(input_sp,(448,448),mode='bilinear', align_corners=False)
# input_sp = input_sp[:,:,0:224,0:224]
#vis
out1=cam_vis(model,input_tensor,name)
sp_cam,mask=sp_cam_vis(model,input_sp,name,out1)
# camp_vis(sp_cam,Q,input_tensor,mask)
camp_vis(sp_cam,Q,input_sp.squeeze(0),mask)






