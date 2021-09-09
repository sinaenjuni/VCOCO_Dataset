
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sb
from pycocotools.coco import COCO
import skimage.io as io

import cv2
from PIL import Image
pd.set_option('display.max_columns', None)

class VCOCOandCOCO:
    def __init__(self, VCOCO_img_path, VCOCO_ann_path, COCO_ann_path, save_path=None):
        self.VCOCO_img_path = VCOCO_img_path
        self.VCOCO_ann_path = VCOCO_ann_path
        self.COCO_ann_path = COCO_ann_path

        self.VCOCO_img_list = self.getFileList(self.VCOCO_img_path, '*')
        self.VCOCO_ann = self.load_json_file(self.VCOCO_ann_path)
        self.COCO_ann = COCO(self.COCO_ann_path)

        print("Find image:", len(self.VCOCO_img_list))
        print("Find annotation", len(self.VCOCO_ann))

        self.cats_ids = self.COCO_ann.getCatIds()
        self.cats = self.COCO_ann.loadCats(self.cats_ids)
        self.save_path = Path(save_path)

    def getFileList(self, path: str, key: str) -> list:
        path = Path(path)
        return list(path.glob(key))

    def load_json_file(self, path):
        if type(path) is list:
            ret = {}
            for ipath in path:
                with open(Path(ipath), 'r') as f:
                    j = json.load(f)
                    ret.update(j)
            return ret
        else:
            with open(Path(path), 'r') as f:
                return json.load(f)

    def getMM2XY(self, bb_mm):
        xmin, ymin, xmax, ymax = bb_mm
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    def getXY2MM(self, bb_xy):
        xmin, ymin, xmax, ymax = bb_xy
        return (xmin, ymin, xmax + xmin, ymax + ymin)

    def getIOU(self, B1, B2):
        xminB1, yminB1, xmaxB1, ymaxB1 = B1
        xminB2, yminB2, xmaxB2, ymaxB2 = B2

        box1_area = (xmaxB1 - xminB1 + 1) * (ymaxB1 - yminB1 + 1);
        box2_area = (xmaxB2 - xminB2 + 1) * (ymaxB2 - yminB2 + 1);

        x1 = max(xminB1, xminB2)
        y1 = max(yminB1, yminB2)
        x2 = min(xmaxB1, xmaxB2)
        y2 = min(ymaxB1, ymaxB2)

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou

    def getClassName(self, classID):
        if type(classID) is str:
            classID = int(classID)

        for cat in self.cats:
            if cat['id'] == classID:
                return cat['name']
        return "None"

    def getAnno(self, idx):
        idx = str(idx)
        for key, val in self.VCOCO_ann.items():
            if str(key) == idx:
                return val
        print(f"ID({idx}) is not found")

    def getImg(self, idx):
        img_path = self.VCOCO_img_list[idx]
        img_id = str(img_path).split('_')[-1].split('.')[0].lstrip('0')
        img_id = int(img_id)
        img_info = self.COCO_ann.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        #         print(img, img_info)
        print(file_name)

        #         I = io.imread('{}/{}'.format(dataDir, file_name))/255.0
        I = io.imread(img_path) / 255.0
        print(I.shape)
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    def getSeg(self, idx):
        img_path = self.VCOCO_img_list[idx]
        img_id = str(img_path).split('_')[-1].split('.')[0].lstrip('0')
        img_id = int(img_id)

        img_info = self.COCO_ann.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        print(file_name)

        annIds = self.COCO_ann.getAnnIds(imgIds=img_id, iscrowd=None)
        print(img_path)
        print(annIds)

        anns = self.COCO_ann.loadAnns(annIds)
        #         coco.showAnns(anns)
        #         print(anns[0].size())

        vval = self.getAnno(img_id)
        for val in vval:
            print(val)

        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            print(ann['bbox'], ann['category_id'], self.getClassName(ann['category_id']))
            pixel_value = int(ann['category_id'])
            mask = np.maximum(self.COCO_ann.annToMask(ann) * pixel_value, mask)

        print(np.unique(mask))
        plt.imshow(mask, cmap='jet')
        plt.show()

    def getTargetSeg(self, idx, is_save=False):
        img_path = self.VCOCO_img_list[idx]
        img_id = str(img_path).split('_')[-1].split('.')[0].lstrip('0')
        img_id = int(img_id)

        img_info = self.COCO_ann.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        print(file_name)

        annIds = self.COCO_ann.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.COCO_ann.loadAnns(annIds)

        filtered_3mask = np.zeros((img_info['height'], img_info['width'], 3))
        vccs = self.getAnno(img_id)
        for vcc in vccs:
            print(vcc)
            pseg = None
            oseg = None
            for ann in anns:
                if self.getIOU(self.getXY2MM(ann['bbox']), vcc['person_bbx']) == 1.0:
                    #                     piou = self.getIOU(self.getXY2MM(ann['bbox']), vcc['person_bbx'])
                    pseg = self.COCO_ann.annToMask(ann)
                    pixel_value = int(ann['category_id'])
                    #                     filtered_mask = np.maximum(self.COCO_ann.annToMask(ann)*pixel_value, filtered_mask)
                    #                     filtered_3mask[...,0] = np.maximum(pseg*pixel_value, filtered_3mask[...,0])
                    filtered_3mask[..., 0] = np.maximum(pseg * 255, filtered_3mask[..., 0])
                    print(self.getClassName(ann['category_id']))

                if len(vcc['object']['obj_bbx']) != 0:
                    if self.getIOU(self.getXY2MM(ann['bbox']), vcc['object']['obj_bbx']) == 1.0:
                        #                         oiou = self.getIOU(self.getXY2MM(ann['bbox']), vcc['object']['obj_bbx'])
                        oseg = self.COCO_ann.annToMask(ann)
                        pixel_value = int(ann['category_id'])
                        #                         filtered_mask = np.maximum(self.COCO_ann.annToMask(ann)*pixel_value, filtered_mask)
                        filtered_3mask[..., 1] = np.maximum(oseg * pixel_value, filtered_3mask[..., 1])
                        print(self.getClassName(ann['category_id']))
            print(vcc['Verbs'])
            if pseg is not None and oseg is not None:
                filtered_3mask[..., 2] = np.maximum(pseg * 255, filtered_3mask[..., 2])
                filtered_3mask[..., 2] = np.maximum(oseg * 255, filtered_3mask[..., 2])
                print(vcc['Verbs'], 'interaction')

        plt.imshow(filtered_3mask.astype(np.uint8), cmap='jet')
        plt.axis('off')
        plt.show()

        filtered_mask = np.zeros((img_info['height'],img_info['width']))
        #         for ann in anns:
        # #             print(ann['bbox'])
        #     #         print(getClassName(ann['category_id'], cats))
        #             if "person" == self.getClassName(ann['category_id']):
        #     #             break
        #                 for val in vccs:
        #                     iou = self.getIOU(self.getXY2MM(ann['bbox']), val['person_bbx'])
        #                     if iou == 1.0:
        #                         pixel_value = int(ann['category_id'])
        #                         filtered_mask = np.maximum(self.COCO_ann.annToMask(ann)*pixel_value, filtered_mask)
        #                         print(self.getClassName(ann['category_id']))

        #     #                 print(iou)
        #             else:
        #                 for val in vccs:
        #                     if len(val['object']['obj_bbx']) != 0:
        #                         iou = self.getIOU(self.getXY2MM(ann['bbox']), val['object']['obj_bbx'])
        #                         if iou == 1.0:
        #                             pixel_value = int(ann['category_id'])
        #                             filtered_mask = np.maximum(self.COCO_ann.annToMask(ann)*pixel_value, filtered_mask)
        #                             print(self.getClassName(ann['category_id']))
        #     #                     print(iou)

        if is_save:
            assert not self.save_path is None, 'TO save file, should put in save_path'
            save_path = self.save_path / file_name.replace('.jpg', '.png')
            if not save_path.parent.exists():
                save_path.parent.mkdir(exist_ok=True, parents=True)
            #             np.save(save_path, mask)
            print(filtered_3mask.shape)
            im = Image.fromarray(filtered_3mask.astype(np.uint8))
            im.save(save_path)
        else:
            plt.imshow(filtered_mask, cmap='jet')
            plt.axis('off')
            plt.show()


def getXY2MM(bb_xy):
    xmin, ymin, xmax, ymax = bb_xy

    return [xmin,
            ymin,
            xmax + xmin,
            ymax + ymin]


def getIOU(B1, B2):
    xminB1, yminB1, xmaxB1, ymaxB1 = B1
    xminB2, yminB2, xmaxB2, ymaxB2 = B2

    box1_area = (xmaxB1 - xminB1 + 1) * (ymaxB1 - yminB1 + 1);
    box2_area = (xmaxB2 - xminB2 + 1) * (ymaxB2 - yminB2 + 1);

    x1 = max(xminB1, xminB2)
    y1 = max(yminB1, yminB2)
    x2 = min(xmaxB1, xmaxB2)
    y2 = min(ymaxB1, ymaxB2)

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def getPD(type, save_file, COCO, VCOCO):
    for idx, (k, v) in enumerate(VCOCO.items()):
        print(idx)

        # if idx == 10:
        #     break

        imgID = int(k)
        # print(imgID)

        annIDs = COCO.getAnnIds(imgIds=imgID, iscrowd=None)
        anns = COCO.loadAnns(annIDs)


        for _v in v:
            vcoco_verb = _v['Verbs']
            vcoco_pbbox = _v['person_bbx']
            vcoco_obbox = _v['object']['obj_bbx']

            if len(vcoco_obbox) == 0:
                # print('ann', vcoco_pbbox, vcoco_obbox, vcoco_verb)

                save_file = save_file.append({'type':type,
                                                'imgID': k,
                                                'pbbox': vcoco_pbbox,
                                                'obbox': vcoco_obbox,
                                                'verb': vcoco_verb
                                                }, ignore_index=True)

                continue

            for _ann in anns:
                coco_cat_id = _ann['category_id']
                coco_super_class = coco_classes[coco_cat_id]['supercategory']
                coco_class = coco_classes[coco_cat_id]['name']
                coco_bbox = getXY2MM(_ann['bbox'])


                IOU = getIOU(vcoco_obbox, coco_bbox)
                # print('IOU', IOU)

                if IOU == 1:
                    # print('ann', vcoco_pbbox, vcoco_obbox, vcoco_verb, coco_super_class, coco_class)

                    save_file = save_file.append({'type':type,
                                                  'imgID': k,
                                                    'pbbox': vcoco_pbbox,
                                                    'obbox': vcoco_obbox,
                                                    'verb': vcoco_verb,
                                                    'super_class' : coco_super_class,
                                                    'coco_class' : coco_class
                                                    }, ignore_index=True)

    return save_file


if __name__ == '__main__':
    img_train_path = Path('./data/images/train2014')
    img_train_list = list(img_train_path.glob('*.jpg'))

    img_val_path = Path('./data/images/val2014')
    img_val_list = list(img_val_path.glob('*.jpg'))

    print(len(img_train_list))
    print(len(img_val_list))

    COCO_anno_train_path = Path('./data/annotations/coco/instances_train2014.json')
    COCO_train = COCO(COCO_anno_train_path)

    COCO_anno_test_path = Path('./data/annotations/coco/instances_val2014.json')
    COCO_test = COCO(COCO_anno_test_path)

    coco_classes = COCO_train.cats
    print(len(COCO_train.imgs))
    print(coco_classes)
    # COCO_anno_val_path = Path('./data/annotations/coco/instances_val2014.json')
    # COCO_val = COCO(COCO_anno_val_path)


    VCOCO_anno_train = Path('./data/annotations/vcoco/train_annotations.json')
    with open(VCOCO_anno_train, 'r') as f:
        VCOCO_anno_train_file = json.load(f)
    print(len(VCOCO_anno_train_file))

    VCOCO_anno_val = Path('./data/annotations/vcoco/val_annotations.json')
    with open(VCOCO_anno_val, 'r') as f:
        VCOCO_anno_val_file = json.load(f)
    print(len(VCOCO_anno_val_file))


    VCOCO_anno_test = Path('./data/annotations/vcoco/test_annotations.json')
    with open(VCOCO_anno_test, 'r') as f:
        VCOCO_anno_test_file = json.load(f)
    print(len(VCOCO_anno_test_file))


    VCOCO_all_pd = pd.DataFrame()

    VCOCO_all_pd = getPD('train', VCOCO_all_pd, COCO_train, VCOCO_anno_train_file)
    VCOCO_all_pd = getPD('val', VCOCO_all_pd, COCO_train, VCOCO_anno_val_file)
    VCOCO_all_pd = getPD('test', VCOCO_all_pd, COCO_test, VCOCO_anno_test_file)

    print(VCOCO_all_pd['type'].unique())


    # VCOCO_train_pd = pd.DataFrame()


    # print(VCOCO_train_pd)
    VCOCO_all_pd.to_json('./all_pd.json')



