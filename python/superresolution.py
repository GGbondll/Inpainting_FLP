import torch
import torch.nn as nn
from model import *

class superresolution(nn.Module):
    def __init__(self, w_scale, h_scale, p, alpha, dt=0.01, T = 100):
        super(superresolution, self).__init__()
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.result_save_path = './super_resolution_result'
        self.p = p
        self.alpha = alpha
        self.dt = dt
        self.T = T
        
    def loading_data(self):
        image_path = './super_resolution'
        inpainting_images, masks, inpainting_images_name, masks_name = [], [], [], []
        for filename in os.listdir(image_path):
            img = Image.open(image_path+'/'+filename)
            img = np.array(img)
            sr_img, mask = self.upscale(img)
            inpainting_images.append(sr_img)
            inpainting_images_name.append(filename[:-4])
            masks.append(mask)
            masks_name.append(filename[:-4] + '_inpainting_mask')

        return inpainting_images, masks, inpainting_images_name, masks_name
    
    def upscale(self, image):
        W, H, c = image.shape
        superresolution_image = np.zeros((self.w_scale*W, self.h_scale*H,c))
        mask = np.zeros((self.w_scale*W, self.h_scale*H))
        for channel in range(c):
            for i in range(W):
                for j in range(H):
                    superresolution_image[int(i*self.w_scale), int(j*self.h_scale), channel] = image[i,j,channel]
                    mask[int(i*self.w_scale), int( j*self.h_scale)] = 255
        
        return superresolution_image, mask

    def prosses(self):
        inpainting_images, masks, inpainting_images_name, masks_name = self.loading_data()
        length = len(inpainting_images)
        times = 0
        timepoint = time.time()
        with tqdm(total = length) as pt:
            for i in range (length):
                origin_image = inpainting_images[i]
                origin_image = torch.Tensor(origin_image)
                origin_image = tensor_dim3to4(origin_image)
                origin_image_name = inpainting_images_name[i]

                mask = masks[i]
                mask = torch.Tensor(mask)
                mask_name = masks_name[i]   

                u0 = origin_image
                save_image(tensor_dim4to3(u0), self.result_save_path +'/' + origin_image_name +'_'+ mask_name)
                u0 = u0/255
                origin_image = origin_image 
                tic = time.time()
                uT = self.model(u0, origin_image_name, mask, mask_name)
                toc = time.time()
                times += toc-tic
                uT = getnormalize(uT)


                save_image(uT*255, self.result_save_path +'/' + origin_image_name +'_'+ mask_name + ' inpainting')

                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name)
                pt.set_postfix({ 'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)

    def model(self, u0, ureal_name, mask, mask_name):
        u0 = torch.Tensor(u0)

        u0 = u0.to(device)

        M = int(self.T/self.dt)
        l_w = compute_n(self.alpha)
        print('\nwindow width', l_w)
        W = window(self.alpha,l_w)
        W = W.to(device)
        uT = u0
        inpainting_area = np.array(255-mask,int) 
        inpainting_area = torch.Tensor(inpainting_area)
        inpainting_area = inpainting_area.to(device)

        with torch.no_grad():
            runtime = {'first conv':0, 'second conv':0, 'quality assess':0}
            time_points = [0] * 10
            time_points[0] = time.time()
            with tqdm(total=M) as t:
                for epoch in range(M):
                    FLu = conv2d_Mirr_extension_3dim(uT,W)
                    runtime['first conv'] += timestamp(time_points, 1)
                    Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(self.p-2)),W)
                    runtime['second conv'] += timestamp(time_points, 2)
                    uT = uT - self.dt *self.p *Tspu* (inpainting_area/255)

                    runtime['quality assess'] += timestamp(time_points, 3)


                    runtime['total'] = runtime['first conv'] + runtime['second conv'] + runtime['quality assess']
                    t.set_description('I: '+ ureal_name + ', M: ' + mask_name + f', epoch: {epoch+1}')
                    t.set_postfix({
                                'conv1':f"{100*runtime['first conv']/runtime['total']:.1f}%",
                                'conv2':f"{100*runtime['second conv']/runtime['total']:.1f}%",
                                })
                    t.update(1)
                    time_points[0] = time.time()
        uT = tensor_dim4to3(uT)   
     
        return uT.cpu().detach()

    def eval_model(self,u0, ureal, ureal_name, mask, mask_name):
        u0, ureal = torch.Tensor(u0), torch.Tensor(ureal)

        u0 = u0.to(device)
        ureal = ureal.to(device)

        M = int(self.T/self.dt)
        bestuT, bestloss = u0, 1
        psnrs, ssims, losses = [], [], []
        l_w = compute_n(self.alpha)
        print('\nwindow width', l_w)
        W = window(self.alpha,l_w)
        W = W.to(device)
        uT = u0
        inpainting_area = np.array(255-mask,int) 
        inpainting_area = torch.Tensor(inpainting_area)
        inpainting_area = inpainting_area.to(device)

        wandb.init(
            project="Image inpainting for "+ureal_name,
            config={
            "architecture": "fractional Laplace operator",
            "image": ureal_name,
            "mask": mask_name,
            "epochs": f"{M}",
            }
        ) 

        with torch.no_grad():
            runtime = {'first conv':0, 'second conv':0, 'quality assess':0}
            time_points = [0] * 10
            time_points[0] = time.time()
            with tqdm(total=M) as t:
                for epoch in range(M):
                    FLu = conv2d_Mirr_extension_3dim(uT,W)
                    runtime['first conv'] += timestamp(time_points, 1)
                    Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(self.p-2)),W)
                    runtime['second conv'] += timestamp(time_points, 2)
                    uT = uT - self.dt *self.p *Tspu* (inpainting_area/255)
                    re = 1 if epoch<M-1 else 3
                    res = quality_assess(uT, ureal,re=3)
                    runtime['quality assess'] += timestamp(time_points, 3)

                    wandb.log({'epoch': epoch + 1, 'Loss': res['Loss']})
                    wandb.log({'epoch': epoch + 1, 'PSNR': res['PSNR']})
                    wandb.log({'epoch': epoch + 1, 'SSIM': res['SSIM']})
                    if epoch%500==0:
                        img = wandb.Image(np.uint8(getnormalize(tensor_dim4to3(uT.cpu().detach()))*255), caption= ureal_name + ' '+mask_name+ " inpainting image")
                        wandb.log({"log an image": img})

                    psnrs.append(res['PSNR'])
                    ssims.append(res['SSIM'])
                    losses.append(res['Loss'])

                    if losses[-1] < bestloss:
                        bestloss, bestuT = losses[-1], uT
                    runtime['total'] = runtime['first conv'] + runtime['second conv'] + runtime['quality assess']
                    t.set_description('I: '+ ureal_name + ', M: ' + mask_name + f', epoch: {epoch+1}')
                    t.set_postfix({'PSNR': f"{res['PSNR']:.2f}", 'SSIM': f"{res['SSIM']:.3f}", 'Loss': f"{res['Loss']:.5f}",
                                'runtime': f"{runtime['total']:.1f}s",
                                'conv1':f"{100*runtime['first conv']/runtime['total']:.1f}%",
                                'conv2':f"{100*runtime['second conv']/runtime['total']:.1f}%",
                                })
                    t.update(1)
                    time_points[0] = time.time()
        uT = tensor_dim4to3(uT)   
        bestuT = tensor_dim4to3(bestuT)     
        wandb.finish()            
        return uT.cpu().detach(), psnrs, ssims, bestloss, bestuT.cpu().detach(), losses

    def eval_loaddata(self):
        image_path = './super_resolution'
        real_images, inpainting_images, masks, real_images_name, inpainting_images_name, masks_name  = [], [], [], [], [], []
        for filename in os.listdir(image_path):
            img = Image.open(image_path+'/'+filename)
            real_image = np.array(img)
            W, H, c = real_image.shape
            img = img.resize((W//self.w_scale, H//self.h_scale))
            real_images.append(real_image)
            sr_img, mask = self.upscale(np.array(img))
            inpainting_images.append(sr_img)
            real_images_name.append(filename[:-4])
            masks.append(mask)
            masks_name.append(filename[:-4] + '_inpainting_mask')
            inpainting_images_name.append(filename[:-4] + '_inpainting')

        return real_images, inpainting_images, masks, real_images_name, inpainting_images_name, masks_name        


    def eval(self):
        result_save_path = self.result_save_path
        times = 0
        origin_images, inpainting_images, masks, origin_images_name, inpainting_images_name, masks_name = self.eval_loaddata()
        result = []
        allpsnr = []
        allbestloss = []
        alllosses = []
        length = len(origin_images)
        timepoint = time.time()
        with tqdm(total = length) as pt:
            for i in range (len(origin_images)):
                origin_image = origin_images[i]
                origin_image = torch.Tensor(origin_image)
                origin_image = tensor_dim3to4(origin_image)
                origin_image_name = origin_images_name[i]

                mask = masks[i]
                mask = torch.Tensor(mask)
                mask_name = masks_name[i]   
                print(inpainting_images[i].shape)
                u0 = tensor_dim3to4(torch.Tensor(inpainting_images[i]))
                save_image(tensor_dim4to3(u0), result_save_path +'/' + inpainting_images_name[i] +'_'+ mask_name)
                u0 = u0/255 
                origin_image = origin_image/255  
                tic = time.time()
                uT, psnrs, ssims, bestloss, bestuT, losses = self.eval_model(u0,origin_image, origin_image_name, mask, mask_name)
                toc = time.time()
                times += toc-tic
                uT = getnormalize(uT)
                bestuT = getnormalize(bestuT)
                log('Image: ' + origin_image_name + ', mask: ' + mask_name + f" PSNR={max(psnrs):4f}\n"
                    +f"psnr_destroyed={psnrs[0]:4f}, psnr_inpainting={max(psnrs):.2f}"
                    +f"\nssim_destroyed={ssims[0]:.4f}, ssim_inpainting={max(ssims):.4f}"
                    +f"\nloss_destroyed={losses[0]:.4f}, loss_inpainting={max(losses):.4f}",
                    log='./logs/log_inpainting.log')
                save_image(uT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + ' inpainting')
                save_image(bestuT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + ' best inpainting')
                result.append(bestuT)
                allpsnr.append(psnrs)
                alllosses.append(losses)
                allbestloss.append(bestloss)
                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name)
                pt.set_postfix({ 'PSNR': f"{psnrs[-1]:4f}",
                            'SSIM': f"{ssims[-1]:.5f}", 'Loss': f"{losses[-1]:.5f}",
                            'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)        


if __name__ == '__main__':   
    hr = superresolution(2, 2, 1.9, 1.3, 0.01, 100)
    hr.prosses()
