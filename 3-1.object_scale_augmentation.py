
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

json_file = pd.read_json('./all_vcoco_pd.json')
img_path = Path('./data/images/train2014')

getIMG = lambda x : list(img_path.glob(f'*{x}.jpg'))[0]
getCenter = lambda x : (x[0], x[1], x[2], x[3])
selMin = lambda x, y : x if x < y else y
selMax = lambda x, y : x if x > y else y

train_json_file = json_file[json_file['type'] == 'train']
val_json_file = json_file[json_file['type'] == 'val']
test_json_file = json_file[json_file['type'] == 'test']

def getOutterPoints(obbox, pbbox):
    oxmin, oymin, oxmax, oymax = obbox
    pxmin, pymin, pxmax, pymax = pbbox

    xmin = selMin(oxmin, pxmin)
    ymin = selMin(oymin, pymin)
    xmax = selMax(oxmax, pxmax)
    ymax = selMax(oymax, pymax)

    return (xmin, ymin, xmax, ymax)


def getTargetArea(img, outter, target_size = [400, 400], sample=10):
    width, height = img.size
    xmin, ymin, xmax, ymax = outter
    start_x, start_y, end_x, end_y = 0,0,0,0
    is_x = True
    is_y = True
    w = xmax - xmin
    h = ymax - ymin

    if w < target_size[0]:
        start_x = xmin - (target_size[0] - w)
        end_x = xmin
    else:
        is_x = False
        crop_xmin = xmin
        crop_xmax = xmax
    if h < target_size[1]:
        start_y = ymin - (target_size[1] - h)
        end_y = ymin
    else:
        is_y = False
        crop_ymin = ymin
        crop_ymax = ymax

    if start_x <= 0:
        start_x = 0
    if start_y <= 0:
        start_y = 0
    if end_x <= 0:
        end_x = 1
    if end_y <= 0:
        end_y = 1

    if end_x + target_size[0] >= width:
        end_x = xmin - ((target_size[0] + xmin) - width)
    if end_y + target_size[1] >= height:
        end_y = ymin - ((target_size[1] + ymin) - height)


    ret = []

    # if w > target_size[0]:
    #     start_x = xmin
    #     target_size[0] = xmax
    #
    # if h > target_size[1]:
    #     start_y = ymin
    #     target_size[1] = ymax
    #
    # if w < target_size[0]:
    #     start_x = xmin - (target_size[0] - w)
    #     if start_x < 0:
    #         start_x = 0
    #
    # if h < target_size[1]:
    #     start_y = ymin - (target_size[1] - h)
    #     if start_y < 0:
    #         start_y = 0

    print(int(start_x), int(start_y), int(end_x), int(end_y))
    for i in range(sample):
        if is_x:
            crop_xmin = randrange(int(start_x)-1, int(end_x)+1)
            crop_xmax = crop_xmin + target_size[0]
        if is_y:
            crop_ymin = randrange(int(start_y)-1, int(end_y)+1)
            crop_ymax = crop_ymin + target_size[1]

        ret += [(crop_xmin, crop_ymin, crop_xmax, crop_ymax)]
    return ret



def getCneterPoints(bbox):
    xmin, ymin, xmax, ymax = bbox

    center_x = (xmax - xmin) / 2 + xmin
    center_y = (ymax - ymin) / 2 + ymin

    return (center_x, center_y)


def getTargetBound(center, img, target_size):
    center_x, center_y = center
    width, height = img.size
    half_target_size = target_size//2

    if center_x - half_target_size > 0:
        retxmin = center_x - half_target_size
        retxmax = retxmin + target_size
    else:
        retxmin = 0
        retxmax = target_size

    if center_y - half_target_size > 0:
        retymin = center_y - half_target_size
        retymax = retymin + target_size
    else:
        retymin = 0
        retymax = target_size


    retxmin = int(retxmin)
    retymin = int(retymin)
    retxmax = int(retxmax)
    retymax = int(retymax)

    return (retxmin, retymin, retxmax, retymax)



print(train_json_file['verb'].unique())


target_cats = train_json_file[train_json_file['verb']=='eat_instr']

print(target_cats.columns)

# print(target_cats['imgID'])

# print(getIMG('467411'))


for target in list(target_cats.itertuples()):
    imgID = target.imgID
    coco_class = target.coco_class
    obbox = target.obbox
    pbbox = target.pbbox
    super_class = target.super_class
    verb = target.verb

    print(imgID, verb, coco_class, obbox, pbbox) # minmax

    img = Image.open(getIMG(imgID))
    draw = ImageDraw.Draw(img)
    draw.rectangle((obbox), outline=(255, 0, 0), width=3)
    draw.rectangle((pbbox), outline=(0, 0, 255), width=3)
    outter = getOutterPoints(obbox, pbbox)
    print(outter)
    center = getCneterPoints(outter)
    draw.ellipse((center[0]-5, center[1]-5,  center[0]+5, center[1]+5), fill='red', outline='blue')

    # plt.imshow(img)
    # plt.show()

    target_area = getTargetArea(img, outter)

    # target = getTargetBound(center, img, target_size=400)
    # print(target)





    # croped_img = img.crop(target)

    # croped_img = croped_img.resize((400,400))
    # img = img.resize((400,400))


    # ret_img = np.concatenate([img, croped_img],axis=1)
    # plt.imshow(img)
    # plt.show()

    for i in target_area:
        print(i)
        # croped_img = img.crop(i)
        # plt.imshow(croped_img)
        # plt.show()

    # plt.imshow(croped_img)
    # plt.imshow(ret_img)
    # plt.axis(True)
    # plt.tight_layout()
    # plt.show()



    # for i in range(10):
    #     xmin = randrange(0, outter[3])
    #     ymin = randrange(0, outter[4])
    #     xmax = xmin + 400
    #     ymax = ymin + 400
    # # center_point = pbbox+obbox
    #     croped_img = img.crop((xmin,ymin,xmax,ymax))
    #     plt.imshow(croped_img)
    #     plt.show()

    # print(getIMG(imgID))
