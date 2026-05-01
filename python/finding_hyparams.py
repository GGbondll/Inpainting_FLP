from hyperopt import fmin, tpe, space_eval, Trials, hp, SparkTrials
from utils import *
import pyspark

space = {
    'alpha': hp.uniform('alpha', 0, 2),
    'p': hp.uniform('p', 1, 2)
}


result_save_path = './result'

def testing_model(u0, ureal, ureal_name, mask, mask_name, dt, T, alpha, p):
    u0 = torch.Tensor(u0).to(device)
    ureal = torch.Tensor(ureal).to(device)

    M = int(T/dt)
    bestloss = float('inf')
    losses = []

    l_w = compute_n(alpha)
    print('\nwindow width', l_w)
    W = window(alpha,l_w).to(device)

    uT = u0
    inpainting_area = np.array(255-mask, int)
    inpainting_area = torch.Tensor(inpainting_area/255).to(device)
    L2_loss = nn.MSELoss()

    converged_count = 0

    with torch.no_grad():
        runtime = {'first conv':0, 'second conv':0, 'quality assess':0}
        time_points = [0] * 4
        time_points[0] = time.time()
        with tqdm(total=M) as t:
            for epoch in range(M):
                FLu = conv2d_Mirr_extension_3dim(uT,W)
                runtime['first conv'] += timestamp(time_points, 1)
                Tspu = conv2d_Mirr_extension_3dim(FLu*(torch.add(torch.abs(FLu),1e-9)**(p-2)),W)
                runtime['second conv'] += timestamp(time_points, 2)
                uT = uT - dt *p *Tspu* inpainting_area
                
                runtime['quality assess'] += timestamp(time_points, 3)
                isconvergence = ' '
                if epoch>0:
                    lastloss = losses[-1]
                else:
                    lastloss = 0

                losses.append(L2_loss(ureal, getnormalize(uT)).item())

                if abs(lastloss - losses[-1])<1e-9:
                    isconvergence = 'Converged'
                    converged_count += 1
                    if converged_count > 100:
                        break
                else:
                    converged_count = 0
                bestloss = min(bestloss, losses[-1])
                
                runtime['total'] = runtime['first conv'] + runtime['second conv'] + runtime['quality assess']
                t.set_description('Image: '+ ureal_name + ', Mask: ' + mask_name +  isconvergence + f" alpha={alpha}, p= {p}")
                t.set_postfix({'Loss': f"{losses[-1]:.5f}", 'runtime': f"{runtime['total']:.1f}s"})
                t.update(1)
                time_points[0] = time.time()         
    return bestloss


def objective(space):
    alpha = space['alpha']
    p = space['p']
    dt =0.01
    T = [40,20,20,20]
    masks, origin_images_name, masks_name = dataload()

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
    bestloss = testing_model(u0,origin_image, origin_image_name, mask, mask_name,dt,T[j],alpha,p)
    toc = time.time()
    times += toc-tic
    print(f'it takes {times/60:.4f}')

    log('\nImage: ' + origin_image_name + ', mask: ' + mask_name + ' with '+ f" alpha={alpha}, p= {p}"
        +f"\nloss_inpainting={bestloss:.9f}",
        log='./logs/log_hyparams_finding '+ origin_image_name + mask_name +'.log')
    return bestloss


if __name__ == '__main__': 
    image_names = os.listdir(image_path)
    mask_names = os.listdir(mask_path)
    for i in range(len(image_names)):
        for j in range(len(mask_names)):
            trials = Trials() # Initialize trials object

            best = fmin(
            fn=objective, # Objective Function to optimize
            space=space, # Hyperparameter's Search Space
            algo=tpe.suggest, # Optimization algorithm
            max_evals=200, # Number of optimization attempts
            trials = trials,
            verbose=True
            )
            res = space_eval(space, best)
            print(res)
            log('\nImage: ' + image_names[i] + ', mask: ' + mask_names[j] + ' best hyparams: '+ f" alpha={res['alpha']}, p= {res['p']}\n",
            log='./logs/log_best_hyparams.log')