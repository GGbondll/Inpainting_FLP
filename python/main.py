import scipy.io as scio

from model import *
from datasets import *
from Params import *

result_save_path = 'C:/Users/GGBond/Desktop/new_result/big_mask'#修复图像保存路径


dt = 0.008
T = [60,40,40,40]

def run_result():
    times = 0


    masks, origin_images_name, masks_name = dataload()
    saves = {'image': [], 'mask': [], 'method': [], 'psnr': [], 'ssim': [], 'loss': []}
    length = len(origin_images_name) * len(masks)
    timepoint = time.time()
    with tqdm(total = length) as pt:
        for i in range (len(origin_images_name)):
            for j in range (len(masks)):

                origin_image_name = origin_images_name[i]
                origin_image = loadimage(origin_image_name)
                origin_image_name = origin_image_name[:-4]
                mask = masks[j]
                mask = torch.Tensor(mask)
                mask_name = masks_name[j]   

                u0 = origin_image * (mask/255)
                save_image(tensor_dim4to3(u0), result_save_path +'/' + origin_image_name +'_'+ mask_name)
                u0 = u0/255
                origin_image = origin_image/255   
                
                print('\n',origin_image_name)
                try:
                    alpha = params[origin_image_name]['alpha'][mask_name]
                except:
                    alpha = 1.3
                try:
                    p = params[origin_image_name]['p'][mask_name]
                except:
                    p =1.95
                print(f'\nalpha = {alpha}, p = {p}')
                saves['image'].append(origin_image_name)
                saves['mask'].append(mask_name)
                saves['method'].append('FLP')
                #saves['alpha'].append(alpha)
                #saves['p'].append(p)
                tic = time.time()
                uT, psnrs, ssims, bestloss, bestuT, losses = model1(u0,origin_image, origin_image_name, mask, mask_name,0.01,500,alpha,p,False)
                toc = time.time()
                compute_time = toc-tic
                # saves['time'].append(compute_time)
                times += toc-tic
                uT = getnormalize(uT)
                bestuT = getnormalize(bestuT)
                
                log('\nImage: ' + origin_image_name + ', mask: ' + mask_name + f", alpha = {alpha}, p = {p}" + f", PSNR={max(psnrs):6f}\n"
                    +f"psnr_destroyed={psnrs[0]:6f}, psnr_inpainting={max(psnrs):.6f}"
                    +f"\nssim_destroyed={ssims[0]:.6f}, ssim_inpainting={max(ssims):.6f}"
                    +f"\nloss_destroyed={losses[0]:.6f}, loss_inpainting={min(losses):.6f}",
                    log='./logs/new_result.log')
                
                save_image(uT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + ' inpainting')
                # save_image(bestuT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + ' best inpainting')
                saves['psnr'].append(psnrs[-1])
                saves['ssim'].append(ssims[-1])
                saves['loss'].append(losses[-1])
                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name)
                pt.set_postfix({    'PSNR': f"{psnrs[-1]:4f}",
                                    'SSIM': f"{ssims[-1]:.5f}", 'Loss': f"{losses[-1]:.5f}",
                                    'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)
            write_excel(saves, 'C:/Users/GGBond/Desktop/FLP_result')
        
    return times

def find_best_p_alpha(i,j):
    p = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    alpha = [1e-9,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2-1e-9]
    p = p[:2]
    alpha = alpha[:2]
    allpsnrs, allssims, alllosses = [alpha], [alpha], [alpha]


    masks, origin_images_name, masks_name = dataload()
    length = len(alpha) * len(p)
    timepoint = time.time()

    origin_image_name = origin_images_name[i] 
    origin_image = loadimage(origin_image_name)
    origin_image_name = origin_image_name[:-4]
    mask = masks[j]
    mask = torch.Tensor(mask)
    mask_name = masks_name[j]   
    
    u0 = origin_image * (mask/255)
    u0 = u0/255
    origin_image = origin_image/255   
    times = 0
    tic = time.time()
    result_save_path = './result/alpha&p'
    with tqdm(total = length) as pt:
        for l in range (len(p)):
            temp_psnr = []
            temp_ssim = []
            temp_loss = []
            for k in range (len(alpha)): 

                _, psnrs, ssims, _, _, losses = model1(u0,origin_image, origin_image_name, mask, mask_name,dt,T[j],alpha[k],p[l])
                toc = time.time()
                times += toc-tic
                print(f'it takes {times/60:.4f}')

                log('\nImage: ' + origin_image_name + ', mask: ' + mask_name + ' with '+ f" alpha={alpha[k]}, p= {p[l]}\n"
                    +f"psnr_destroyed={psnrs[0]:6f}, psnr_inpainting={psnrs[-1]:.6f}"
                    +f"\nssim_destroyed={ssims[0]:.6f}, ssim_inpainting={ssims[-1]:.6f}"
                    +f"\nloss_destroyed={losses[0]:.6f}, loss_inpainting={losses[-1]:.6f}",
                    log='./logs/log_find_best_prameter '+ origin_image_name + '.log')

                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name)
                pt.set_postfix({ 'PSNR': f"{psnrs[-1]:4f}",
                                'SSIM': f"{ssims[-1]:.5f}", 'Loss': f"{losses[-1]:.5f}",
                                'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)

                temp_psnr.append(psnrs[-1])
                temp_ssim.append(ssims[-1])
                temp_loss.append(losses[-1])
            allpsnrs.append(temp_psnr)
            allssims.append(temp_ssim)
            alllosses.append(temp_loss)
        write_excel(allssims, result_save_path +'/' + origin_image_name +'_'+ mask_name +'_ssims2')
        write_excel(allpsnrs, result_save_path +'/' + origin_image_name +'_'+ mask_name +'_psnrs2')
        write_excel(alllosses, result_save_path +'/' + origin_image_name +'_'+ mask_name +'_losses2')
    
            
    return 
if __name__ == '__main__':   
    
    a = time.time() 
    times = run_result()
    b = time.time()
    '''

    find_best_p_alpha(0,1)  
    '''