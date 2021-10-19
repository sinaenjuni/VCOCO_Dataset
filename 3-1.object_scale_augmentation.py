
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from random import randrange

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

def getCneterPoints(bbox):
    xmin, ymin, xmax, ymax = bbox

    center_x = (xmax - xmin) / 2 + xmin
    center_y = (ymax - ymin) / 2 + ymin

    return (center_x, center_y)

def getTargetBound(center, img, target_size):
    center_x, center_y = center
    width, height = img.size
    retxmin = center_x - (target_size//2) if center_x - (target_size//2) > 0 else 0
    retymin = center_y - (target_size//2) if center_y - (target_size//2) > 0 else 0
    retxmax = center_x + (target_size//2) if center_x + (target_size//2) > width else width
    retymax = center_y + (target_size//2) if center_y + (target_size//2) > height else height

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


for target in list(target_cats.itertuples())[:1]:
    imgID = target.imgID
    coco_class = target.coco_class
    obbox = target.obbox
    pbbox = target.pbbox
    super_class = target.super_class
    verb = target.verb

    print(imgID, verb, coco_class, obbox, pbbox) # minmax

    img = Image.open(getIMG(imgID))

    outter = getOutterPoints(obbox, pbbox)
    center = getCneterPoints(outter)
    target = getTargetBound(center, img, target_size=400)

    for i in range(10):
        xmin = randrange(target[0], target[2])
        ymin = randrange(target[1], target[3])
        xmax = xmin + 400
        ymax = ymin + 400
        print(xmin, ymin, xmax, ymax)
        # print(xmax - xmin, ymax - ymin)
        croped_img = img.crop((xmin,ymin,xmax,ymax))
        plt.imshow(croped_img)
        plt.show()

    draw = ImageDraw.Draw(img)
    draw.rectangle((obbox), outline=(255, 0, 0), width=3)
    draw.rectangle((pbbox), outline=(0, 0, 255), width=3)

    draw.ellipse((center[0]-5, center[1]-5,  center[0]+5, center[1]+5), fill='red', outline='blue')

    plt.imshow(img)
    plt.show()



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
