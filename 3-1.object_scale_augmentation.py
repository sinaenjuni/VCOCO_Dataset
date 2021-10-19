
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

def getTargetPoints(img, bbox, target_size=400):
    xmin, ymin, xmax, ymax = bbox

    print(img.size)
    print(xmin, ymin, xmax, ymax)
    w=(xmax-xmin)
    h=(ymax-ymin)
    for i in range(sample):
        x1 = randrange(0, x - matrix)
        y1 = randrange(0, y - matrix)
        sample_list.append(img.crop((x1, y1, x1 + matrix, y1 + matrix)))

    if w < target_size:
        margin_w = target_size - w
        xmin = xmin - margin_w/2
        xmax = xmax + margin_w/2

    if h < target_size:
        margin_h = target_size - h
        ymin = ymin - margin_h/2
        ymax = ymax + margin_h/2

    print(xmin, ymin, xmax, ymax)
    return (xmin, ymin, xmax, ymax)



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
    draw = ImageDraw.Draw(img)
    draw.rectangle((obbox), outline=(255, 0, 0), width=3)
    draw.rectangle((pbbox), outline=(0, 0, 255), width=3)

    plt.imshow(img)
    plt.show()

    outter = getOutterPoints(obbox, pbbox)

    outter = getTargetPoints(img, outter, target_size=400)


    # center_point = pbbox+obbox
    croped_img = img.crop(outter)
    plt.imshow(croped_img)
    plt.show()

    # print(getIMG(imgID))
