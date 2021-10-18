
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from PIL import Image
from pathlib import Path
import numpy as np

base_path = Path('save_files/fork')

img_path = base_path / 'img'
seg_path = base_path / 'seg'

imgs = img_path.glob('*.png')
imgs = list(imgs)


from fractions import Fraction


def crop_resize(image, size, ratio):
    print(ratio)
    # crop to ratio, center
    w, h = image.size
    if w > ratio * h: # width is larger then necessary
        x, y = (w - ratio * h) // 2, 0
    else: # ratio*height >= width (height is larger)
        x, y = 0, (h - w / ratio) // 2
    image = image.crop((x, y, w - x, h - y))

    # resize
    if image.size > size: # don't stretch smaller images
        image.thumbnail(size, Image.ANTIALIAS)
    return image



for img_path in imgs[:5]:
    ret = []

    scaling = 1.3

    img_name = img_path.name.replace('.png', '')
    seg_names = seg_path.glob(f'{img_name}*')
    segs = [np.array(Image.open(seg_path)) for seg_path in seg_names]


    img = Image.open(img_path)
    img_ret = np.array(img)

    img_width, img_height = img.size
    img_scaled_width, img_scaled_height = int(img_width*scaling), int(img_height*scaling)

    img_crop_start_width_point, img_crop_start_height_point = abs(img_scaled_width-img_width)//2, abs(img_scaled_height-img_height)//2

    # print(img_width, img_height)
    # print(img_scaled_width, img_scaled_height)
    # print((img_scaled_width-img_width)//2, (img_scaled_height-img_height)//2)

    # img = np.array(img)
    # img_ori = np.pad(img_ori, (2,2), mode='constant')

    # img_resized = Image.new(img.mode, (img_width*2, img_height*2), (0, 0, 255))
    # img_resized.paste(img, (img_width//2, img_height//2))


    # img = img.resize((640,427), box=(10,10, 200, 200))

    plt.imshow(img)
    plt.axis(False)
    plt.show()

    seg_rets = []
    for seg in segs:
        seg = (seg//255)[:,:,None]
        seg = np.repeat(seg, 3, axis=-1)
        seg = img * seg
        seg = Image.fromarray(seg)

        seg = seg.resize((img_scaled_width, img_scaled_height), resample=Image.BICUBIC)

        # plt.imshow(seg)
        # plt.show()
        # print('awdwad', seg.size)
        # seg = crop_resize(seg, (img_width, img_height), Fraction(img_width, img_height))
        # print(seg.size)

        # print(Fraction(img_scaled_width, img_scaled_height))

        # plt.imshow(seg)
        # plt.show()
        seg = seg.crop((img_crop_start_width_point, img_crop_start_height_point, img_width+img_crop_start_width_point, img_height+img_crop_start_height_point))
        # seg = seg.crop((0, 0, img_width, img_height))
        seg = np.array(seg)
        seg_mask = np.where(seg == 0, 1, 0)

        img_ret = img_ret * seg_mask
        # print(np.unique(seg_mask))

        # plt.imshow(seg_mask)
        # plt.show()
        seg_rets += [seg]


    for seg_ret in seg_rets:
        img_ret += seg_ret
        # print(seg_ret.size)

    plt.imshow(img_ret)
    plt.axis(False)
    plt.show()



