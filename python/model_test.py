from utils import *
from Params import *

result_save_path = 'C:/Users/GGBond/Desktop/test_image/result'

dt = 0.01
T = [60,40,40,40]
l = 2

def data_load():
    mypath_gt = 'C:/Users/GGBond/Desktop/test_image/noise'
    mypath_mask = 'C:/Users/GGBond/Desktop/test_image/mask'
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


def main(l):
    times = 0

    masks, origin_images_name, masks_name = data_load()
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
                u0 = u0/255
                origin_image = origin_image/255   
                tic = time.time()
                print('\n',origin_image_name)
                for char in ['building','cat','fox','face','rabbit','penguin','forest']:
                    if char in origin_image_name:
                        gt_name = char
                alpha = params[gt_name]['alpha'][mask_name] if gt_name in params else 1.3
                p = params[gt_name]['p'][mask_name] if gt_name in params else 1.95
                print(f'\nalpha = {alpha}, p = {p}')
                gt = load_gt(origin_image_name)
                uT, psnrs, ssims, bestloss, bestuT, losses = model_test(u0, gt/255, origin_image_name, mask, mask_name, dt, T[j], alpha, p, l)
                toc = time.time()
                times += toc-tic
                log('\nImage: ' + origin_image_name + ', mask: ' + mask_name + f", alpha = {alpha}, p = {p}, lambda = {l}" + f", PSNR={max(psnrs):6f}\n"
                    +f"psnr_destroyed={psnrs[0]:6f}, psnr_inpainting={max(psnrs):.6f}"
                    +f"\nssim_destroyed={ssims[0]:.6f}, ssim_inpainting={max(ssims):.6f}"
                    +f"\nloss_destroyed={losses[0]:.6f}, loss_inpainting={min(losses):.6f}",
                    log='./logs/denoise.log')
                save_image(uT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + f' inpainting_lambda{l}')
                save_image(bestuT*255, result_save_path +'/' + origin_image_name +'_'+ mask_name + f' best_inpainting_lambda{l}')
                pt.set_description('image: '+ origin_image_name + ', mask:' + mask_name + f'lambda = {l}')
                pt.set_postfix({ 'PSNR': f"{psnrs[-1]:4f}",
                               'SSIM': f"{ssims[-1]:.5f}", 'Loss': f"{losses[-1]:.5f}",
                               'runtime': f'{(time.time()-timepoint):.2f}s',
                                })
                pt.update(1)
    
        
    return times

def model_test(u0, ureal, ureal_name, mask, mask_name, dt, T, alpha, p, l):
    u0, ureal = torch.Tensor(u0).to(device), torch.Tensor(ureal).to(device)
    omega = torch.zeros_like(u0).to(device)

    M = int(T/dt)
    bestuT, bestloss = u0, float('inf')
    psnrs, ssims, losses = [], [], []
    l_w = compute_n(alpha)
    print('\nwindow width', l_w)
    W = window(alpha,l_w).to(device)
    uT = u0
    h, w = mask.shape
    inpainting_area = np.array(255-mask,int) 
    inpainting_area = torch.Tensor(inpainting_area/255).bool().reshape(1,1,h,w).to(device)
    converged_count = 0

    with torch.no_grad():
        runtime = {'first conv':0, 'second conv':0, 'quality assess':0}
        time_points = [0] * 10
        time_points[0] = time.time()
        with tqdm(total=M) as t:
            for epoch in range(M):
                FLu = conv2d_Mirr_extension_3dim(uT,W)
                runtime['first conv'] += timestamp(time_points, 1)
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                L_omega = Laplace(omega)
                runtime['second conv'] += timestamp(time_points, 2)
                uT = uT + dt * (-p * Tspu - omega)
                omega = omega + dt * (L_omega + l * (uT - u0))
                omega = omega.masked_fill(inpainting_area,0)
                re = 3 if converged_count>0 or epoch == 0 or epoch == M-1 else 1
                res = quality_assess(uT, ureal, re=3)
                runtime['quality assess'] += timestamp(time_points, 3)
                isconvergence = ' '
                if epoch>0:
                    lastloss = losses[-1]
                else:
                    lastloss = 0

                psnrs.append(res['PSNR'])
                ssims.append(res['SSIM'])
                losses.append(res['Loss'])

                if abs(lastloss - losses[-1])<1e-9:
                    isconvergence = 'Converged'
                    converged_count += 1
                    if converged_count > M:
                        break
                else:
                    converged_count = 0

                if losses[-1] < bestloss:
                    bestloss, bestuT = losses[-1], uT

                runtime['total'] = runtime['first conv'] + runtime['second conv'] + runtime['quality assess']
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name + f', epoch: {epoch+1} '+ isconvergence)
                t.set_postfix({'PSNR': f"{res['PSNR']:.2f}", 
                               'SSIM': f"{res['SSIM']:.3f}", 
                               'Loss': f"{res['Loss']:.9f}",
                               'runtime': f"{runtime['total']:.1f}s"
                              })
                t.update(1)
                time_points[0] = time.time()
    uT = tensor_dim4to3(uT)   
    bestuT = tensor_dim4to3(bestuT)             
    return uT.cpu().detach(), psnrs, ssims, bestloss, bestuT.cpu().detach(), losses

def test(u0, ureal, mask, dt, T, alpha, p, l):
    u0, ureal = torch.Tensor(u0).to(device), torch.Tensor(ureal).to(device)
    omega = torch.zeros_like(u0).to(device)
    h, w = mask.shape
    M = int(T/dt)
    l_w = compute_n(alpha)
    print('\nwindow width', l_w)
    W = window(alpha,l_w).to(device)
    uT = u0

    inpainting_area = np.array(255-mask,int) 
    save_image(inpainting_area, result_save_path +'/' + 'inpainting_area')

    inpainting_area = torch.Tensor(inpainting_area/255).bool().reshape(1,1,h,w).to(device)

    print('\ninpainting area shape: ', inpainting_area.shape, torch.max(inpainting_area))

    with torch.no_grad():
        for epoch in range(M):
            FLu = conv2d_Mirr_extension_3dim(uT,W)
            Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
            L_omega = Laplace(omega)
            uT = uT + dt * (-p * Tspu - omega)
            save_image(getnormalize(tensor_dim4to3(uT).cpu().detach())*255, result_save_path +'/' + 'uT')
            
            save_image(tensor_dim4to3((uT - u0)).cpu().detach()*255, result_save_path +'/' + 'not_masked')
            save_image(tensor_dim4to3((uT - u0).masked_fill(inpainting_area,0)).cpu().detach()*255, result_save_path +'/' + 'outside_of_masked_area')
            
            omega = omega + dt * (L_omega + l * (uT - u0))
            save_image(tensor_dim4to3(omega).cpu().detach()*255, result_save_path +'/' + 'omega1')
            omega = omega.masked_fill(inpainting_area,0)
            save_image(tensor_dim4to3(omega).cpu().detach()*255, result_save_path +'/' + 'omega2')
            print('\nuT shape, max uT', uT.shape, torch.max(uT))
            print('\nomega shape, max omega', omega.shape, torch.max(omega))
            res = quality_assess(uT, ureal, re=3)
            print('\n', res)

def run_test():
    masks, origin_images_name, masks_name = data_load()
    i, j = 0, 0 
    origin_image = loadimage(origin_images_name[i])
    origin_image_name = origin_images_name[i][:-4]
    mask = masks[j]
    mask = torch.Tensor(mask)
    mask_name = masks_name[j]   

    u0 = origin_image * (mask/255)
    u0 = u0/255
    origin_image = origin_image/255   
    tic = time.time()
    print('\n',origin_image_name)
    for char in ['building','cat','fox','face','rabbit','penguin','forest']:
        if char in origin_image_name:
            gt_name = char
    alpha = params[gt_name]['alpha'][mask_name] if gt_name in params else 1.3
    p = params[gt_name]['p'][mask_name] if gt_name in params else 1.95
    print(f'\nalpha = {alpha}, p = {p}')
    gt = load_gt(origin_image_name)
    test(u0, gt/255, mask, dt, T[j], alpha, p, l)

if __name__ == '__main__':
    for lam in [0.5, 1, 2, 5,10]:
        main(lam)
    pass