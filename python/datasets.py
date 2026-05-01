from utils import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
import numpy as np

class inpainting_data(Dataset):
    def __init__(self, path = './gt'):
        super(inpainting_data, self).__init__()
        self.path = path
        self.name = os.listdir(self.path)
        self.length = len(self.name)

    def dataload(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)
        return img
    
    def __getitem__(self, index):
        data = {}
        data['name'] = self.name[index][:-4]
        image = self.dataload(os.path.join(self.path, self.name[index]))
        image = torch.Tensor(image)
        image = tensor_dim3to4(image)
        data['image'] = image
        return data

    def __len__(self):
        return self.length

class mask_Data(Dataset):
    def __init__(self, path = './mask'):
        super(mask_Data, self).__init__()
        self.path = path
        self.name = os.listdir(self.path)
        self.length = len(self.name)

    def __len__(self):
        return self.length
    
    def dataload(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)
        return img
    def __getitem__(self, index):
        data = {}
        data['name'] = self.name[index][:-4]
        mask = self.dataload(os.path.join(self.path, self.name[index]))
        mask = torch.Tensor(mask)
        data['mask'] = mask
        return data

if __name__ == '__main__':   
    image_data = inpainting_data('./matlab/resize/gt')
    mask_data = mask_Data('./matlab/resize/mask')
    image_loader = DataLoader(image_data, batch_size=1, shuffle=False)
    mask_loader = DataLoader(mask_data, batch_size=1, shuffle=False)
    times = 0
    result = []
    allpsnr = []
    allbestloss = []
    alllosses = []
    length = image_data.length * mask_data.length
    timepoint = time.time()
    with tqdm(total = length) as pt:
        for i , images in enumerate(tqdm(image_loader)):
            origin_image_name = images['name'][0]
            origin_image = images['image'][0]
            for j , masks in enumerate(tqdm(mask_loader)):
                mask = masks['mask'][0]
                mask_name = masks['name'][0]
                print(origin_image_name, origin_image.shape)
                print(mask_name, mask.shape)
            
    


