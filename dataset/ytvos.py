from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainYTVOS(torch.utils.data.Dataset):
    def __init__(self, root, split, clip_n):
        self.root = root
        self.split = split
        with open(os.path.join(root, 'ImageSets', '{}.txt'.format(split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, self.split, 'JPEGImages', video_name)
        flow_dir = os.path.join(self.root, self.split, 'JPEGFlows', video_name)
        mask_dir = os.path.join(self.root, self.split, 'Annotations', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # select training frame
        all_frames = list(range(len(img_list)))
        frame_id = random.choice(all_frames)
        img = Image.open(img_list[frame_id]).convert('RGB')
        flow = Image.open(flow_list[frame_id]).convert('RGB')
        mask = Image.open(mask_list[frame_id]).convert('P')

        # resize to 512p
        img = img.resize((512, 512), Image.BICUBIC)
        flow = flow.resize((512, 512), Image.BICUBIC)
        mask = mask.resize((512, 512), Image.NEAREST)

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
        masks = self.to_mask(mask).unsqueeze(0)
        masks = (masks != 0).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks}
