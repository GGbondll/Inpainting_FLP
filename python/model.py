from utils import *
import wandb

def model1(u0, ureal, ureal_name, mask, mask_name, dt, T, alpha, p, wandb_off = True):
    u0 = torch.Tensor(u0).to(device)
    ureal = torch.Tensor(ureal).to(device)

    M = int(T/dt)
    bestuT, bestloss = u0, float('inf')
    psnrs, ssims, losses = [0], [0], [0]
    l_w = compute_n(alpha)
    print('\nwindow width', l_w)
    W = window(alpha,l_w).to(device)
    uT = u0
    inpainting_area = np.array(255-mask,int)
    inpainting_area = torch.Tensor(inpainting_area/255).to(device)
    converged_count = 0
    if not wandb_off:
        wandb.init(
            project="Image inpainting for "+ureal_name+' model1',
            config={
            "architecture": "fractional Laplace operator",
            "image": ureal_name,
            "mask": mask_name,
            "T": f"{T}",
            "dt": f"{dt}",
            "alpha": f'{alpha}',
            "p": f'{p}'
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
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                runtime['second conv'] += timestamp(time_points, 2)
                uT = uT - dt *p *Tspu* inpainting_area
                re = 3 if converged_count>0 or epoch == 0 or epoch == M-1 else 1
                res = quality_assess(uT, ureal,3)
                runtime['quality assess'] += timestamp(time_points, 3)
                isconvergence = ' '

                if not wandb_off:
                    wandb.log({'epoch': epoch + 1, 'Loss': res['Loss']})
                    wandb.log({'epoch': epoch + 1, 'PSNR': res['PSNR']})
                    wandb.log({'epoch': epoch + 1, 'SSIM': res['SSIM']})
                                
                    if epoch%1000==0:
                        img = wandb.Image(np.uint8(getnormalize(tensor_dim4to3(uT.cpu().detach()))*255), caption= ureal_name + ' '+mask_name+ " inpainting image")
                        wandb.log({"log an image": img})
                                    

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
                    if converged_count > 500:
                        break
                else:
                    converged_count = 0

                if losses[-1] < bestloss:
                    bestloss, bestuT = losses[-1], uT
                
                runtime['total'] = runtime['first conv'] + runtime['second conv'] + runtime['quality assess']
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name + f' alpha: {alpha}, p: {p}, '+ isconvergence)
                t.set_postfix({'PSNR': f"{res['PSNR']:.2f}", 
                               'SSIM': f"{res['SSIM']:.3f}", 
                               'Loss': f"{res['Loss']:.9f}",
                               'runtime': f"{runtime['total']:.1f}s"
                              })
                t.update(1)
                time_points[0] = time.time()
    uT = tensor_dim4to3(uT)   
    bestuT = tensor_dim4to3(bestuT)     
    if not wandb_off:
        wandb.finish()         
    return uT.cpu().detach(), psnrs, ssims, bestloss, bestuT.cpu().detach(), losses

def model2(u0, ureal, ureal_name, mask, mask_name, dt, T, alpha, p, l, wandb_off = True):
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
    if not wandb_off:
        wandb.init(
            project="Image inpainting and denoising for "+ ureal_name+' model2',
            config={
            "architecture": "fractional Laplace operator",
            "image": ureal_name,
            "mask": mask_name,
            "T": f"{T}",
            "dt": f"{dt}",
            "alpha": f'{alpha}',
            "p": f'{p}',
            "lambda": f'{l}'
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
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                L_omega = Laplace(omega)
                runtime['second conv'] += timestamp(time_points, 2)
                uT = uT + dt * (-p * Tspu - omega)
                omega = omega + dt * (L_omega + l * (uT - u0))
                omega = omega.masked_fill(inpainting_area,0)
                re = 3 if converged_count>0 or epoch == 0 or epoch == M-1 else 1
                res = quality_assess(uT, ureal, re=3)
                runtime['quality assess'] += timestamp(time_points, 3)
                if not wandb_off:
                    wandb.log({'epoch': epoch + 1, 'Loss': res['Loss']})
                    wandb.log({'epoch': epoch + 1, 'PSNR': res['PSNR']})
                    wandb.log({'epoch': epoch + 1, 'SSIM': res['SSIM']})
                    if epoch%500==0:
                        img = wandb.Image(np.uint8(tensor_dim4to3(uT.cpu().detach())*255), caption= ureal_name + ' '+mask_name+ " inpainting image")
                        wandb.log({"log an image": img})

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
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name + f'alpha: {alpha}, p: {p}, lambda: {l} '+ isconvergence)
                t.set_postfix({'PSNR': f"{res['PSNR']:.2f}", 
                               'SSIM': f"{res['SSIM']:.3f}", 
                               'Loss': f"{res['Loss']:.9f}",
                               'runtime': f"{runtime['total']:.1f}s"
                              })
                t.update(1)
                time_points[0] = time.time()
    uT = tensor_dim4to3(uT)   
    bestuT = tensor_dim4to3(bestuT)     
    if not wandb_off:
        wandb.finish()            
    return uT.cpu().detach(), psnrs, ssims, bestloss, bestuT.cpu().detach(), losses


def model_l2(u0, ureal, ureal_name, mask, mask_name, dt, T, alpha, p, l, wandb_off = True):
    u0, ureal = torch.Tensor(u0).to(device), torch.Tensor(ureal).to(device)

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
    if not wandb_off:
        wandb.init(
            project="Image inpainting and denoising for "+ ureal_name+' model2',
            config={
            "architecture": "fractional Laplace operator",
            "image": ureal_name,
            "mask": mask_name,
            "T": f"{T}",
            "dt": f"{dt}",
            "alpha": f'{alpha}',
            "p": f'{p}',
            "lambda": f'{l}'
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
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                runtime['second conv'] += timestamp(time_points, 2)
                omega = uT - u0
                omega = omega.masked_fill(inpainting_area,0)
                uT = uT + dt * (-p * Tspu - l * omega)
                re = 3 if converged_count>0 or epoch == 0 or epoch == M-1 else 1
                res = quality_assess(uT, ureal, re=3)
                runtime['quality assess'] += timestamp(time_points, 3)
                if not wandb_off:
                    wandb.log({'epoch': epoch + 1, 'Loss': res['Loss']})
                    wandb.log({'epoch': epoch + 1, 'PSNR': res['PSNR']})
                    wandb.log({'epoch': epoch + 1, 'SSIM': res['SSIM']})
                    if epoch%500==0:
                        img = wandb.Image(np.uint8(tensor_dim4to3(uT.cpu().detach())*255), caption= ureal_name + ' '+mask_name+ " inpainting image")
                        wandb.log({"log an image": img})

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
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name + f', alpha: {alpha}, p: {p}, lambda: {l} '+ isconvergence)
                t.set_postfix({'PSNR': f"{res['PSNR']:.2f}", 
                               'SSIM': f"{res['SSIM']:.3f}", 
                               'Loss': f"{res['Loss']:.9f}",
                               'runtime': f"{runtime['total']:.1f}s"
                              })
                t.update(1)
                time_points[0] = time.time()
    uT = tensor_dim4to3(uT)   
    bestuT = tensor_dim4to3(bestuT)     
    if not wandb_off:
        wandb.finish()            
    return uT.cpu().detach(), psnrs, ssims, bestloss, bestuT.cpu().detach(), losses

def inpainting(u0, ureal_name, mask, mask_name, dt, T, alpha, p):
    u0 = torch.Tensor(u0).to(device)

    M = int(T/dt)
    bestuT, bestloss = u0, float('inf')
    l_w = compute_n(alpha)
    W = window(alpha,l_w).to(device)
    uT = u0
    inpainting_area = torch.Tensor(np.array(255-mask,int)/255).to(device)
    converged_count = 0
    losses = []
    with torch.no_grad():
        with tqdm(total=M) as t:
            for epoch in range(M):
                FLu = conv2d_Mirr_extension_3dim(uT,W)
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                last_u = uT
                uT = uT - dt *p *Tspu* inpainting_area
                cross_loss = l2_loss(uT, last_u).item()
                isconvergence = ' '

                if epoch>0:
                    lastloss = losses[-1]
                else:
                    lastloss = 0
               
                losses.append(cross_loss)
                
                if abs(lastloss - losses[-1])<1e-9:
                    isconvergence = 'Converged'
                    converged_count += 1
                    if converged_count > 10:
                        break
                else:
                    converged_count = 0

                if losses[-1] < bestloss:
                    bestloss, bestuT = losses[-1], uT
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name + f'alpha: {alpha}, p: {p}, '+ isconvergence)
                t.set_postfix({'Loss': f"{cross_loss:.9f}"})
                t.update(1)
    uT = tensor_dim4to3(uT)   
    bestuT = tensor_dim4to3(bestuT)     
     
    return uT.cpu().detach(), bestloss, bestuT.cpu().detach(), losses