from utils import *
import pandas as pd
alpha = 1.5
n = 2


mask = torch.Tensor([[255,0,255,0,0],[0,0,0,0,0],[0,0,23,0,0],[0,0,0,0,31],[0,0,0,0,0]])


test = torch.Tensor(
    [[[[1,2,3],[4,5,6],[7,8,9]]],
     [[[1,2,3],[4,5,6],[7,8,9]]],
     [[[1,2,3],[4,5,6],[7,8,9]]]]
)

image = torch.Tensor([[[1,2,3,4,5],
                      [6,7,8,9,10],
                      [11,12,13,14,15],
                      [16,17,18,19,20],
                      [21,22,23,24,25]],
                      [[1,2,3,4,5],
                      [6,7,8,9,10],
                      [11,12,13,14,15],
                      [16,17,18,19,20],
                      [21,22,23,24,25]],
                      [[1,2,3,4,5],
                      [6,7,8,9,10],
                      [11,12,13,14,15],
                      [16,17,18,19,20],
                      [21,22,23,24,25]]
                      ])

mask = torch.BoolTensor([[1,0,0,0,0],
                         [1,0,0,0,0],
                         [1,0,0,0,0],
                         [1,0,0,0,0],
                         [1,0,0,0,0]])


def mask_process(mask):
    mask = mask[0][0]
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 1)
    mask = mask.byte()
    return mask

def func():
    import pandas as pd
    file_path = 'C:/Users/GGBond/Desktop/alpha&p/result2/'
    saves = {'image name': [], 'mask': [], 'method': [], 'psnr': [], 'ssim': [], 'loss': [],'alpha': [], 'p': []}
    for name in os.listdir(file_path):
        df = pd.read_excel(file_path+name).astype(float).values[:,1:]
        df = np.array(df)
        if 'psnr' in name:
            saves['image name'].append(name[:15])
            for mask_name in ['mark', 'random', 'scratch', 'waterprint']:
                if mask_name in name:
                    saves['mask'].append(mask_name)
            max_value = np.max(df)
            saves['psnr'].append(max_value)
            max_index = np.unravel_index(np.argmax(df), df.shape)
            alpha = max_index[1]/10+0.9
            p = max_index[0]/10 + 1.0
            saves['alpha'].append(alpha)
            saves['p'].append(p)
            saves['method'] = 'FLP'
        if 'ssim' in name:
            saves['ssim'].append(np.max(df))
        if 'loss' in name:
            saves['loss'].append(np.min(df))
    write_excel(saves, 'C:/Users/GGBond/Desktop/alpha_and_p_3')
def func2():
    a=[1,2]
    b=[3,4,5]
    res=map(lambda x,y :x+y,a,b)
    print(list(res))

if __name__ == '__main__':
    func2()
    pass