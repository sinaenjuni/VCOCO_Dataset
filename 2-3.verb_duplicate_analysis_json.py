
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


def getRmDuplicate(json_file, ind):
    ret = {}
    for _ind in ind:
        tverb = json_file[json_file['verb'] == _ind]
        # non_duplicate = tverb.drop_duplicates(subset='imgID')
        non_duplicate = tverb
        ret[_ind] = len(non_duplicate)
    return pd.Series(ret)


if __name__ == '__main__':
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
    output_file = pd.DataFrame(index=ind)

    train_json_file = json_file[json_file['type']=='train']
    val_json_file = json_file[json_file['type']=='val']
    test_json_file = json_file[json_file['type']=='test']
    trainval_json_file = pd.concat([train_json_file, val_json_file])


    output_file['Train'] = getRmDuplicate(train_json_file, ind)
    output_file['Val'] = getRmDuplicate(val_json_file, ind)
    output_file['Test'] = getRmDuplicate(test_json_file, ind)
    output_file['Trainval'] = getRmDuplicate(trainval_json_file, ind)
    print(output_file)



    # for _ind in ind:
    #     tverb = val_json_file[val_json_file['verb']==_ind]
    #     non_duplicate = tverb.drop_duplicates(subset='imgID')
    #     print( _ind, len(non_duplicate) )
    #
    # for _ind in ind:
    #     tverb = test_json_file[test_json_file['verb']==_ind]
    #     non_duplicate = tverb.drop_duplicates(subset='imgID')
    #     print( _ind, len(non_duplicate) )
    #
    #
    # for _ind in ind:
    #     tverb = trainval_json_file[trainval_json_file['verb']==_ind]
    #     non_duplicate = tverb.drop_duplicates(subset='imgID')
    #     print( _ind, len(non_duplicate) )
    #
    #
    # print(train_json_file)
    # print(val_json_file)
    # print(trainval_json_file)

    # output_file['Train'] = getVC(train_json_file, 'verb')
    # output_file['Val'] = getVC(val_json_file, 'verb')
    # output_file['Test'] = getVC(test_json_file, 'verb')
    # output_file['Trainval'] = getVC(trainval_json_file, 'verb')
    #
    # output_file['Num_train_verb'] = getVerbCounts(train_json_file)
    # output_file['Num_val_verb'] = getVerbCounts(val_json_file)
    # output_file['Num_test_verb'] = getVerbCounts(test_json_file)
    # output_file['Num_trainval_verb'] = getVerbCounts(trainval_json_file)
    #
    # output_file['List_train_verb'] = getVerbList(train_json_file)
    # output_file['List_val_verb'] = getVerbList(val_json_file)
    # output_file['List_test_verb'] = getVerbList(test_json_file)
    # output_file['List_trainval_verb'] = getVerbList(trainval_json_file)
    #
    # output_file['Num_train_verb_per_classes'] = output_file['Train'] / output_file['Num_train_verb']
    # output_file['Num_val_verb_per_classes'] = output_file['Val'] / output_file['Num_val_verb']
    # output_file['Num_test_verb_per_classes'] = output_file['Test'] / output_file['Num_test_verb']
    # output_file['Num_trainval_verb_per_classes'] = output_file['Trainval'] / output_file['Num_trainval_verb']




    # print(output_file)
