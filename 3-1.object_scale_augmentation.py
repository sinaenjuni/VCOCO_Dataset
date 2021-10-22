
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
    xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox = outter
    width_bbox = xmax_bbox - xmin_bbox
    height_bbox = ymax_bbox - ymin_bbox

    if width_bbox <= target_size[0] and width > target_size[0]:
        x_min = xmin_bbox - (target_size[0] - width_bbox)
        x_max = xmin_bbox + target_size[0]

        x_start = 0 if x_min < 0 else x_min
        x_end = xmin_bbox if x_max < width else xmin_bbox - (x_max - width)  +1

        x_start = int(x_start)
        x_end = int(x_end)

        if x_start == x_end:
            x_min_crop = [x_start for i in range(sample)]

        else:
            x_min_crop = [randrange(x_start, x_end) for i in range(sample)]
        x_max_crop = [i + target_size[0] for i in x_min_crop]

    else:
        x_min_crop = [xmin_bbox]
        x_max_crop = [xmax_bbox]


    if height_bbox <=  target_size[1] and height > target_size[1]:
        y_min = ymin_bbox - (target_size[1] - height_bbox)
        y_max = ymin_bbox + target_size[1]

        y_start = 0 if y_min < 0 else y_min
        y_end = ymin_bbox if y_max < height else ymin_bbox - (y_max - height) +1

        y_start = int(y_start)
        y_end = int(y_end)

        if y_start == y_end:
            y_min_crop = [y_start for i in range(sample)]
        else:
            y_min_crop = [randrange(y_start, y_end) for i in range(sample)]
        y_max_crop = [i + target_size[1] for i in y_min_crop]
    else:
        y_min_crop = [ymin_bbox]
        y_max_crop = [ymax_bbox]

    ret = []
    for xmin, ymin, xmax, ymax in zip(x_min_crop, y_min_crop, x_max_crop, y_max_crop):
        ret += [(xmin, ymin, xmax, ymax)]

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


for target in list(target_cats.itertuples())[:4]:
    imgID = target.imgID
    coco_class = target.coco_class
    obbox = target.obbox
    pbbox = target.pbbox
    super_class = target.super_class
    verb = target.verb

    # print(imgID, verb, coco_class, obbox, pbbox) # minmax

    img = Image.open(getIMG(imgID))
    draw = ImageDraw.Draw(img)
    draw.rectangle((obbox), outline=(255, 0, 0), width=3)
    draw.rectangle((pbbox), outline=(0, 0, 255), width=3)
    outter = getOutterPoints(obbox, pbbox)
    center = getCneterPoints(outter)
    draw.ellipse((center[0]-5, center[1]-5,  center[0]+5, center[1]+5), fill='red', outline='blue')

    plt.imshow(img)
    plt.show()

    target_area = getTargetArea(img, outter, target_size=(400,400),sample=5)

    for i in target_area:
        # print(i)
        croped_img = img.crop(i)
        plt.imshow(croped_img)
        plt.show()

    # target = getTargetBound(center, img, target_size=400)
    # print(target)





    # croped_img = img.crop(target)

    # croped_img = croped_img.resize((400,400))
    # img = img.resize((400,400))


    # ret_img = np.concatenate([img, croped_img],axis=1)
    # plt.imshow(img)
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
