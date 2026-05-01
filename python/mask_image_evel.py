from utils import *
import time

def evel_masked_image():
    masks, origin_images_name, masks_name = dataload()
    result = []
    allpsnr = []
    allbestloss = []
    alllosses = []
    length = len(origin_images_name) * len(masks)
    result = []
    with tqdm(total = length) as pt:
        with torch.no_grad():
            for i in range (len(origin_images_name)):
                for j in range (len(masks)):
                    origin_image_name = origin_images_name[i]
                    origin_image = loadimage(origin_image_name)
                    origin_image_name = origin_image_name[:-4]
                    mask = masks[j]
                    mask = torch.Tensor(mask)
                    mask_name = masks_name[j]   
                    u0 = origin_image * (mask/255)
                    res = quality_assess(u0.to(device)/255,origin_image.to(device)/255)
                    result.append([origin_image_name, mask_name, 'mask image', res['PSNR'], res['SSIM'].item()])
            write_excel(result, './' + 'mask image')

def compute_window_size():
    alpha = [1e-9,1e-5,0.01,0.05,0.06,0.07,0.08,0.09, 0.1,0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2-1e-9]
    p = [1+1e-9, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2-1e-9]
    result = [alpha]
    with torch.no_grad():
        cat_image = loadimage('cat.png').to(device)
        for _ in range(100):
            tic = time.time()
            using_time = []
            for a in alpha:
                l_w = compute_n(a)
                W = window(a,l_w).to(device)
                t1 = time.time()
                for _ in range(100):
                    FLu = conv2d_Mirr_extension_3dim(cat_image,W)
                    Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(1.9-2)),W)  
                t2 = time.time()
                using_time.append((t2-t1))
            result.append(using_time)
            toc = time.time()
            print(f'{toc-tic:.5f}s')
            write_excel(result, './'+'window size and times3')
    write_excel(result, './'+'window size and times3')

if __name__ == '__main__':
    compute_window_size()
    pass