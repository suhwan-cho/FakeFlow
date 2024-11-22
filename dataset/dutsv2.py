from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDUTSv2(torch.utils.data.Dataset):
    def __init__(self, root, clip_n):
        self.root = root
        img_dir = os.path.join(root, 'JPEGImages')
        flow_dir = os.path.join(root, 'JPEGFlows')
        mask_dir = os.path.join(root, 'Annotations')
        self.img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        all_frames = list(range(len(self.img_list)))
        frame_id = random.choice(all_frames)
        img = Image.open(self.img_list[frame_id]).convert('RGB')
        flow = Image.open(self.flow_list[frame_id]).convert('RGB')
        mask = Image.open(self.mask_list[frame_id]).convert('L')

        # resize to 512p
        img = img.resize((512, 512), Image.BICUBIC)
        flow = flow.resize((512, 512), Image.BICUBIC)
        mask = mask.resize((512, 512), Image.BICUBIC)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            flow = TF.hflip(flow)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            flow = TF.vflip(flow)
            mask = TF.vflip(mask)

        # convert formats
        imgs = self.to_tensor(img).unsqueeze(0)
        flows = self.to_tensor(flow).unsqueeze(0)
        masks = self.to_tensor(mask).unsqueeze(0)
        masks = (masks > 0.5).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks}
