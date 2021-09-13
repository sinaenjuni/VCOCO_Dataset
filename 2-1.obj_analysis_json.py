
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

    json_file = pd.read_json('./all_vcoco_pd.json')
    # output_file = pd.DataFrame(index=ind)

    train_json_file = json_file[json_file['type']=='train']
    val_json_file = json_file[json_file['type']=='val']
    test_json_file = json_file[json_file['type']=='test']
    trainval_json_file = pd.concat([train_json_file, val_json_file])

    output_file = pd.DataFrame(index=trainval_json_file['coco_class'].unique())

    print(len(train_json_file['coco_class'].unique()))
    print(len(val_json_file['coco_class'].unique()))
    print(len(test_json_file['coco_class'].unique()))
    print(len(trainval_json_file['coco_class'].unique()))


    train_json_file['area'] = train_json_file['obbox'].apply(area)
    val_json_file['area'] = val_json_file['obbox'].apply(area)
    test_json_file['area'] = test_json_file['obbox'].apply(area)
    trainval_json_file['area'] = trainval_json_file['obbox'].apply(area)


    output_file['Num_train_classes'] = pd.Series(train_json_file['coco_class'].value_counts())
    output_file['Train_area'] = getObjMeanArea(train_json_file)

    output_file['Num_val_classes'] = val_json_file['coco_class'].value_counts()
    output_file['Val_area'] = getObjMeanArea(val_json_file)


    output_file['Num_test_classes'] = test_json_file['coco_class'].value_counts()
    output_file['Test_area'] = getObjMeanArea(test_json_file)


    output_file['Num_trainval_classes'] = trainval_json_file['coco_class'].value_counts()
    output_file['Trainval_area'] = getObjMeanArea(trainval_json_file)




    print(output_file)
    # output_file.to_csv('./all_vcoco_obj.csv')

    # verb_counts(train_json_file)
    # verb_counts(val_json_file)
    # verb_counts(test_json_file)


    # print('True')