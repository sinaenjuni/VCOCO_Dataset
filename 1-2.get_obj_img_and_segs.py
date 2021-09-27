from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
import requests
import numpy as np
import torch
from torchvision.utils import save_image


import pandas as pd
pd.set_option('display.max_columns', None)


def getVC(pd, key):
    return pd[key].value_counts()


def getVerbCounts(json_file):
    ret = pd.Series()
    verb_list = json_file['verb'].unique()
    for verb in verb_list:
        ret[verb] = len(json_file[json_file['verb'] == verb]['coco_class'].unique())

    return ret

def getVerbList(json_file):
    ret = pd.Series()
    verb_list = json_file['verb'].unique()
    for verb in verb_list:
        ret[verb] = json_file[json_file['verb'] == verb]['coco_class'].unique()

    return ret


# def getVerbList(json_file):
#     verb_count = json_file['verb'].value_counts()
#     verb_list = json_file['verb'].unique()
#
#     # print(verb_count)
#     # print(verb_list)
#
#     pd_verb_count = pd.DataFrame(verb_count)
#
#     obj_list = pd.Series()
#     obj_list_num = pd.Series()
#     for verb in verb_list:
#         objs = json_file[json_file['verb'] == verb]['coco_class'].unique()
#
#         obj_list[verb] = objs
#         obj_list_num[verb] = len(objs)
#
#     # print(obj_list)
#     pd_verb_count['objs'] = obj_list
#     pd_verb_count['num_objs'] = obj_list_num
#         # print(verb, len(obj_list))
#
#
#     return pd_verb_count
    # pd_verb_count = pd_verb_count.reindex(ind)
    # print(pd_verb_count)
    # pd_verb_count.to_csv('./train_verb_counts.csv')

def getObjMeanArea(json_file):
    objs = json_file['coco_class'].unique()

    ret = pd.Series()
    for obj in objs:
        ret[obj] = json_file[json_file['coco_class']==obj]['area'].mean()

    return ret


if __name__ == '__main__':
    img_train_path = Path('./data/images/train2014')
    img_train_list = list(img_train_path.glob('*.jpg'))

    # img_val_path = Path('./data/images/val2014')
    # img_val_list = list(img_val_path.glob('*.jpg'))

    print(len(img_train_list))
    # print(len(img_val_list))

    COCO_anno_train_path = Path('./data/annotations/coco/instances_train2014.json')
    COCO_train = COCO(COCO_anno_train_path)

    # COCO_anno_test_path = Path('./data/annotations/coco/instances_val2014.json')
    # COCO_test = COCO(COCO_anno_test_path)


    eat_instr_objs = ['spoon', 'fork', 'knife']
    coco_classes = {v['name'] : k for k, v in COCO_train.cats.items() if v['name'] in eat_instr_objs}

    print(coco_classes)

    # print(True if 48 in coco_classes.values() else False)

    area = lambda x: (x[2] - x[0]) * (x[3] - x[1]) if len(x) != 0 else []
    ind = ['skateboard',
           'surf',
           'snowboard',
           'hit_instr',
           'ride',
           'kick',
           'work_on_computer',
           'talk_on_phone',
           'drink',
           'jump',
           'ski',
           'hit_obj',
           'hold',
           'cut_instr',
           'throw',
           'catch',
           'read',
           'look',
           'carry',
           'eat_obj',
           'cut_obj',
           'sit',
           'lay',
           'eat_instr']

    json_file = pd.read_json('./all_coco_pd.json')

    # output_file = pd.DataFrame(index=ind)

    train_json_file = json_file[json_file['type']=='train']
    test_json_file = json_file[json_file['type']=='test']
    print('num train obj', len(train_json_file))
    print('num test obj', len(test_json_file))

    # trainval_json_file = pd.concat([train_json_file, val_json_file])
    #
    # output_file = pd.DataFrame(index=trainval_json_file['coco_class'].unique())
    #

    # print(COCO_train.imgs[531378])
    # print(COCO_train.loadImgs(531378))
    # print(train_json_file[train_json_file['coco_class'] == 'spoon'])

    SAVE =  Path('./save_files')

    for obj in eat_instr_objs[2:3]:
        print(obj)
        obj_id = coco_classes[obj]
        cls = train_json_file[train_json_file['coco_class']==obj]
        img_ids = cls.drop_duplicates(subset=['imgID'])

        print(obj, len(img_ids), len(cls))

        for ind_img, img_id in enumerate(img_ids['imgID']):
            img_id = int(img_id)
            print(ind_img, img_id)
            img_info = COCO_train.loadImgs(img_id)[0]
            img = Image.open(requests.get(img_info['coco_url'], stream=True).raw)

            IMG_SAVE = SAVE / obj / 'img' / f'{ind_img}.png'
            if not IMG_SAVE.parent.exists():
                IMG_SAVE.parent.mkdir(exist_ok=True, parents=True)
            img.save(IMG_SAVE)

            # import matplotlib.pyplot as plt
            # plt.imshow(img)

            anno_ids = COCO_train.getAnnIds(img_id)

            annos = COCO_train.loadAnns(anno_ids)
            annos = [anno for anno in annos if anno['category_id'] == obj_id]
            # print([anno['category_id'] for anno in annos])
            # print(anno_ids)
            # COCO_train.showAnns(annos)
            # plt.show()

            for ind_seg, anno in enumerate(annos):
                img_seg = COCO_train.annToMask(anno)*255
                img_seg = Image.fromarray(img_seg)

                SEG_SAVE = SAVE / obj / 'seg' / f'{ind_img}_{ind_seg}.png'
                if not SEG_SAVE.parent.exists():
                    SEG_SAVE.parent.mkdir(exist_ok=True, parents=True)
                img_seg.save(SEG_SAVE)
                # plt.imshow(img_seg)
                # plt.show()



    # print(len(train_json_file['coco_class']))
    # print(len(test_json_file['coco_class']))
    # print(len(trainval_json_file['coco_class'].unique()))
    #
    #
    # train_json_file['area'] = train_json_file['obbox'].apply(area)
    # val_json_file['area'] = val_json_file['obbox'].apply(area)
    # test_json_file['area'] = test_json_file['obbox'].apply(area)
    # trainval_json_file['area'] = trainval_json_file['obbox'].apply(area)
    #
    #
    # output_file['Num_train_classes'] = pd.Series(train_json_file['coco_class'].value_counts())
    # output_file['Train_area'] = getObjMeanArea(train_json_file)
    #
    # output_file['Num_val_classes'] = val_json_file['coco_class'].value_counts()
    # output_file['Val_area'] = getObjMeanArea(val_json_file)
    #
    #
    # output_file['Num_test_classes'] = test_json_file['coco_class'].value_counts()
    # output_file['Test_area'] = getObjMeanArea(test_json_file)
    #
    #
    # output_file['Num_trainval_classes'] = trainval_json_file['coco_class'].value_counts()
    # output_file['Trainval_area'] = getObjMeanArea(trainval_json_file)
    #
    #
    #
    #
    # print(output_file)
    # output_file.to_csv('./all_vcoco_obj.csv')

    # verb_counts(train_json_file)
    # verb_counts(val_json_file)
    # verb_counts(test_json_file)


    # print('True')

# import matplotlib.pyplot as plt
# if __name__ == '__main__':
#     PATH = Path('./save_files')
#     for dir in PATH.iterdir():
#         for type in dir.iterdir():
#             for file in type.iterdir():
#                 if type.name == 'img':
#                     img = Image.open(file).convert('RGB')
#                     plt.imshow(img)
#                     plt.title(str(file))
#                     plt.show()
#                 else:
#                     seg = Image.open(file).convert('L')
#                     n_seg = np.array(seg)
#                     print(n_seg.min(), n_seg.max())
#                     plt.imshow(seg)
#                     plt.title(str(file))
#                     plt.show()

