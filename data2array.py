from keras.preprocessing.image import load_img, img_to_array
import pickle
import os

# base_path = 'D:/lyb/'
base_path = '/Users/mahaoyang/Downloads/'


def data2array(path):
    with open(path + 'DatasetA_train_20180813/label_list.txt', 'r') as f:
        label_list = dict()
        for line in f:
            line = line.strip('\n').split('\t')
            label_list[line[0]] = line[1]
    print('label_list', len(label_list))

    if not os.path.exists('train_list.pickle'):

        with open(path + 'DatasetA_train_20180813/train.txt', 'r') as f:
            train_list = dict()
            for line in f:
                line = line.strip('\n').split('\t')
                train_list[line[0]] = dict()
                train_list[line[0]]['lable'] = line[1]

        for img in train_list:
            pic = load_img(path + 'DatasetA_train_20180813/train/' + img, target_size=(64, 64))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            train_list[img]['array'] = pic

        with open('train_list.pickle', 'wb') as f:
            pickle.dump(train_list, f)

    else:

        with open('train_list.pickle', 'rb') as f:
            train_list = pickle.load(f)
    print('train_list', len(train_list))

    with open(path + 'DatasetA_train_20180813/class_wordembeddings.txt', 'r') as f:
        class_wordembeddings = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            class_wordembeddings[int(ii[0])] = ii[1:]
    print('class_wordembeddings', len(class_wordembeddings))

    with open(path + 'DatasetA_train_20180813/attributes_per_class.txt', 'r') as f:
        attributes_per_class = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attributes_per_class[ii[0]] = ii[1:]
    print('attributes_per_class', len(attributes_per_class))

    with open(path + 'DatasetA_train_20180813/attribute_list.txt', 'r') as f:
        attribute_list = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attribute_list[int(ii[0])] = ii[1:]
    print('attribute_list', len(attribute_list))

    for i in train_list:
        train_list[i]['label_real_name'] = label_list[i]
        train_list[i]['label_real_name_class_wordembeddings'] = class_wordembeddings[label_list[i]]
        train_list[i]['label_attribute'] = attributes_per_class[label_list[i]]

    return {'lable_list': label_list, 'train_list': train_list, 'attributes_per_class': attributes_per_class,
            'attribute_list': attribute_list}


if __name__ == '__main__':
    data2array(base_path)
