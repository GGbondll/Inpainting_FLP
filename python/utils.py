import torch
import math
import torch.nn as nn
import os
import PIL.Image as Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
import pytorch_ssim
import cv2
from mpl_toolkits.mplot3d import Axes3D
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = './matlab/resize/gt'
mask_path = './matlab/resize/mask'
print('\nusing device:', device)
def timestamp(time_points, n):
    time_points[n] = time.time()
    return time_points[n] - time_points[n-1]

def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')
    else:
        pass
    if str:
        return string+end

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', log=True, last_epoch=0):
        self.name = name
        self.fmt = fmt
        self.log = log
        self.history = []
        self.last_epoch = last_epoch
        self.history_init_flag = False
        self.reset()

    def reset(self):
        if self.log:
            try:
                if self.avg>0: self.history.append(self.avg)
            except:
                pass#print(f'Start log {self.name}!')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        try:
            k = len(val)
            self.count += k
            self.avg = self.avg + (sum(val) - self.avg) / self.count
            self.sum += sum(val)
        except:
            self.count += 1
            self.avg = self.avg + (val-self.avg)/self.count
            self.sum += val
    
    def plot_history(self, savefile='log.jpg', logfile='log.pkl'):
        # 读取老log
        if os.path.exists(logfile) and not self.history_init_flag:
            self.history_init_flag = True
            with open(logfile, 'rb') as f:
                history_old = pickle.load(f)
                if self.last_epoch: # 为0则重置
                    self.history = history_old + self.history[:self.last_epoch]
        # 记录log
        with open(logfile, 'wb') as f:
            pickle.dump(self.history, f)
        # 画图
        plt.figure(figsize=(12,9), dpi = 200)
        plt.title(f'{self.name} log')
        x = list(range(len(self.history)))
        plt.plot(x, self.history)
        plt.xlabel('Epoch')
        plt.ylabel(self.name)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def average(self):
        return self.avg

def quality_assess(X, Y, re=3, norm = True):
    if X.shape != Y.shape:
        print(f'X shape{X.shape} is different to Y.shape{Y.shape}')
    if norm:
        X = getnormalize(X)
    psnr = 0
    ssim = 0
    if re > 1:
        psnr = PSNR(X, Y)
    if re > 2:
        ssim = SSIM(X.permute(1,0,2,3), Y.permute(1,0,2,3))
    loss = l2_loss(Y,X)
    if re > 2:
        return {'PSNR':psnr.item(), 'SSIM': ssim.item(), 'Loss': loss.item()}
    elif re >1:
        return {'PSNR':psnr.item(), 'SSIM': 0.0, 'Loss': loss.item()}
    else:
        return {'PSNR':0.0, 'SSIM': 0.0, 'Loss': loss.item()}
def l2_loss(gt,u):
    loss = nn.MSELoss()
    return loss(gt, u)

def PSNR(res,gt):
    mse = torch.mean((res - gt) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def SSIM(img1, img2, window_size = 11, size_average = True):
    device = img1.device
    (_, channel, _, _) = img1.size()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def window(alpha,n):
    '''
    alpha is parameter of fractional laplace operator
    n is Window radius
    '''
    n = int(n)
    s = alpha/2
    w = np.zeros((2*n+1, 2*n+1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i == n and j == n:
                continue
            else:
                coefficient = ((n-i)**2 + (n-j)**2)**(-1-s)
                w[i, j] = -coefficient
                w[2*n-i, j] = -coefficient
                w[i, 2*n-j] = -coefficient
                w[2*n - i, 2*n-j] = -coefficient
                
    w[n,n] = -np.sum(w)
    w = 4**s*math.gamma(1+s)/(torch.pi*abs(math.gamma(-s))) * w
    return torch.FloatTensor(w)


def Error_n(n, alpha):
    sums = sum(1/( (n+1)**2 + k**2 )**(1+alpha/2) for k in range(1,n+1) )
    e = 8*sums+4*( 1/(n+1)**(2+alpha) + 1/(2*(n+1)**2)**(1+alpha/2) )
    return e
    
    
def compute_n(alpha):
    c = 2**alpha*math.gamma(1+alpha/2)/(math.pi*abs(math.gamma(-alpha/2)))
    e = c*Error_n(1,alpha)
    n=1
    while 510*e>=0.5:
        n+=1
        e=c*Error_n(n,alpha)
    return int(n)
    
def conv2d_Mirr_extension_3dim(image,mask):
    m1, n2 = mask.shape[0], mask.shape[1]
    m, n = mask.shape[0]//2, mask.shape[1]//2
    M, N = image.shape[2], image.shape[3]
    result = torch.zeros(3,1,M+2*m,N+2*n,device=device)
    for channel in range(3):
        image_channel = image[channel, 0, :, :]

        
        leftup = torch.rot90(torch.rot90(image_channel[1:m+1,1:n+1]))
        leftdown =torch.rot90(torch.rot90(image_channel[M-m-1:M-1,1:n+1]))
        rightup = torch.rot90(torch.rot90(image_channel[1:m+1,N-n-1:N-1]))
        rightdown = torch.rot90(torch.rot90(image_channel[M-m-1:M-1,N-n-1:N-1]))

        Mirr_extension_image = torch.concatenate((torch.concatenate((leftup,torch.flip(image_channel[1:m+1,:], dims=[0]),rightup), axis=1)
                                            , torch.concatenate((torch.flip(image_channel[:,1:m+1], dims=[1]), image_channel, torch.flip(image_channel[:,N-n:N], dims=[1])), axis=1)
                                            , torch.concatenate((leftdown,torch.flip(image_channel[M-m:M,:], dims=[0]),rightdown), axis=1)
                                            ), axis=0).view(M+2*m,N+2*n)
        result[channel, 0 :, :] = Mirr_extension_image

    mask = mask.view(1,1,m1,n2)

    return F.conv2d(result, mask)

def conv2d_Mirr_extension(image,mask):
    m1, n2 = mask.shape[0], mask.shape[1]
    m, n = mask.shape[0]//2, mask.shape[1]//2
    M, N = image.shape[0], image.shape[1]
    leftup = torch.rot90(torch.rot90(image[1:m+1,1:n+1]))
    leftdown =torch.rot90(torch.rot90(image[M-m-1:M-1,1:n+1]))
    rightup = torch.rot90(torch.rot90(image[1:m+1,N-n-1:N-1]))
    rightdown = torch.rot90(torch.rot90(image[M-m-1:M-1,N-n-1:N-1]))


    Mirr_extension_image = torch.concatenate((torch.concatenate((leftup,torch.flip(image[1:n+1,:], dims=[0]),rightup), axis=1)
                                           , torch.concatenate((torch.flip(image[:,1:m+1], dims=[1]),image, torch.flip(image[:,M-m:M], dims=[1])), axis=1)
                                           , torch.concatenate((leftdown,torch.flip(image[N-n:N,:], dims=[0]),rightdown), axis=1)
                                           ), axis=0).view(1,M+2*m,N+2*n)

    mask = mask.view(1,1,m1,n2)
    conv=F.conv2d(Mirr_extension_image,mask)
    
    return conv[0,:,:]

def tensor_dim3to4(x):
    if len(x.shape) < 3:
        x = x.unsqueeze(dim = 2)
    m, n, c = x.shape
    x = x.view(1, m, n, c)
    x = x.permute(3, 0, 1, 2)
    return x

def tensor_dim4to3(x):
    y = x[:, 0, :, :]
    y = y.permute(1, 2, 0)
    if y.shape[2] == 1:
        y = y[:,:,0]
    return y

def dataload():
    mypath_gt = 'C:/Users/GGBond/Desktop/new_result/gt'
    mypath_mask = 'C:/Users/GGBond/Desktop/new_result/mask'
    path_gt = mypath_gt if mypath_gt else image_path
    path_mask = mypath_mask if mypath_mask else mask_path

    masks, origin_images_name, masks_name = [], [], []
    for filename in os.listdir(path_gt):
        origin_images_name.append(filename)

    for filename in os.listdir(path_mask):
        img = Image.open(path_mask+'/'+filename)
        masks.append(np.array(img))   
        masks_name.append(filename[:-4])

    return masks, origin_images_name, masks_name

def save_image(uT, name):
    uT = np.uint8(uT)
    img = Image.fromarray(uT)
    img.save(name +'.png')

def draw_3d_image(x,y,z, name):
    fig = plt.figure(figsize=(10, 10), facecolor='white', dpi =400) #创建图片
    sub = fig.add_subplot(111, projection='3d')# 添加子图，
    surf = sub.plot_surface(x, y, z, cmap=plt.cm.brg) #绘制曲面,cmap=plt.cm.brg并设置颜色cmap
    cb = fig.colorbar(surf, shrink=0.8, aspect=15) #设置颜色棒
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")

    plt.savefig(name + '.png')

def getnormalize(u):
    u = (u - torch.min(u))/(torch.max(u) - torch.min(u)+1e-9)
    return u

def Laplace(u):
    kernel = torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]])
    kernel = kernel.to(device)
    return conv2d_Mirr_extension_3dim(u,kernel)

def repaint(uT, u0, mask):
    result = torch.where(mask == 0, uT, u0)
    return result

def loadimage(filename, mask = False):
    mypath_gt = 'C:/Users/GGBond/Desktop/new_result/gt'
    path_gt = mypath_gt if mypath_gt else image_path
    img = Image.open(path_gt+'/'+filename)
    img = np.array(img)
    img = torch.Tensor(img)
    if not mask:
        img = tensor_dim3to4(img)
    return img

def load_gt(filename):
    gt_path = 'C:/Users/GGBond/Desktop/new_result/gt'
    for char in ['building','cat','fox','face','rabbit','penguin','forest']:
        if char in filename:
            name = char+'.png'
    img = Image.open(gt_path+'/'+name)
    img = np.array(img)
    img = torch.Tensor(img)
    img = tensor_dim3to4(img)
    return img

def write_excel(data,name):
    import pandas as pd
    datafrme = pd.DataFrame(data)
    datafrme.to_excel(name + ".xlsx")

def merge_dicts(dict1, dict2):
    merged_dict = {}

    for key, value in dict1.items():
        merged_dict[key] = value.copy()
    
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value.copy()
            
    return merged_dict