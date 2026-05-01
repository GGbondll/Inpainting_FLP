from model import *
from datasets import *
from Params import *


result_save_path = 'C:/Users/GGBond/Desktop/success_image/additive_noise'


dt = 0.008
T = [60,40,40,40]

def run_result():
    times = 0

    masks, origin_images_name, masks_name = dataload()
    saves = {'image': [], 'mask': [], 'lambda': [], 'psnr': [], 'ssim': [], 'loss': [],'alpha': [], 'p': []}
    length = len(origin_images_name) * len(masks)
    timepoint = time.time()
    
    with tqdm(total = length) as pt:
        for image_name  in [item for item in origin_images_name if 'rabbit' in item]:
            for j in range (len(masks)):

                origin_image_name = image_name
                origin_image = loadimage(origin_image_name)
                origin_image_name = origin_image_name[:-4]
                
                mask = masks[j]
                mask = torch.Tensor(mask)
                mask_name = masks_name[j]   
                

                u0 = origin_image * (mask/255)
                save_image(tensor_dim4to3(u0), result_save_path +'/noise_mask/' + origin_image_name +'_'+ mask_name)
                u0 = u0/255
                origin_image = origin_image/255   
                tic = time.time()
                print('\n',origin_image_name)
                for char in ['building','cat','fox','face','rabbit','penguin','forest']:
                    if char in origin_image_name:
                        gt_name = char
                alpha = params[gt_name]['alpha'][mask_name] if gt_name in params else 1.3
                p = params[gt_name]['p'][mask_name] if gt_name in params else 1.95
                sigma = origin_image_name[-2:] if origin_image_name[-2:] in ['08','36'] else '08'
                l = noise_l[sigma][gt_name]['lambda'][mask_name] if gt_name in params else 1.95
                saves['image'].append(origin_image_name)
                saves['mask'].append(masks_name)
                saves['lambda'].append(l)
                saves['alpha'].append(alpha)
                saves['p'].append(p)
                print(f'\nalpha = {alpha}, p = {p}, lambda = {l}')
                gt = load_gt(origin_image_name)
                uT, psnrs, ssims, bestloss, bestuT, losses = model2(u0, gt/255, origin_image_name, mask, mask_name, dt, T[j], alpha, p, l, wandb_off= False)
                toc = time.time()
                times += toc-tic
                '''
                uT = getnormalize(uT)
                bestuT = getnormalize(bestuT)
                '''
                log('\nImage: ' + origin_image_name + ', mask: ' + mask_name + f", alpha = {alpha}, p = {p}, lambda = {l}" + f", PSNR={max(psnrs):6f}\n"
                    +f"psnr_destroyed={psnrs[0]:6f}, psnr_inpainting={max(psnrs):.6f}"
                    +f"\nssim_destroyed={ssims[0]:.6f}, ssim_inpainting={max(ssims):.6f}"
                    +f"\nloss_destroyed={losses[0]:.6f}, loss_inpainting={min(losses):.6f}",
                    log='./logs/denoise_result_allnoise.log')
                saves['psnr'].append(max(psnrs))
                saves['ssim'].append(max(ssims))
                saves['loss'].append(min(losses))
                # save_image(uT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + f' inpainting_lambda{l}')
                save_image(bestuT*255, result_save_path +'/denoise_image/' + origin_image_name +'_'+ mask_name + f' inpainting')
                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name + f' alpha = {alpha}, p = {p}, lambda = {l}')
                pt.set_postfix({'PSNR': f"{psnrs[-1]:4f}",
                               'SSIM': f"{ssims[-1]:.5f}", 'Loss': f"{losses[-1]:.5f}",
                               'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)
    
        
    return saves

if __name__ == '__main__':  
    a = time.time()
    saves = run_result()
    b = time.time()
    print(b-a)
    write_excel(saves, result_save_path+'/denoise_result')