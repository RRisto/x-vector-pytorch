import os
import glob

class_ids = {'et': 0, 'ru': 1, 'fi': 2}


def create_meta(files_list, store_loc, mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)

    if mode == 'train':
        meta_store = store_loc + '/training.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'test':
        meta_store = store_loc + '/testing.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'validation':
        meta_store = store_loc + '/validation.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    else:
        print('Error in creating meta files')


def extract_files(folder_path, train_nums=3):
    all_lang_folders = sorted(glob.glob(folder_path + '/*/'))
    train_lists = []
    test_lists = []
    val_lists = []

    for lang_folderpath in all_lang_folders:
        language = lang_folderpath.split('\\')[-2]  # for windows \\

        all_files = sorted(glob.glob(lang_folderpath + '/*.wav'))
        for audio_filepath in all_files[:train_nums]:
            to_write = audio_filepath + ' ' + str(class_ids[language])
            train_lists.append(to_write)

        for audio_filepath in all_files[train_nums:]:
            to_write = audio_filepath + ' ' + str(class_ids[language])
            val_lists.append(to_write)

        for audio_filepath in all_files[train_nums:]:
            to_write = audio_filepath + ' ' + str(class_ids[language])
            test_lists.append(to_write)

    return train_lists, test_lists, val_lists


if __name__ == '__main__':
    class Conf:
        processed_data = 'data/raw/'
        meta_store_path = 'meta_et_ru_fi/'

    config = Conf()
    train_list, test_list, val_lists = extract_files(config.processed_data)

    create_meta(train_list, config.meta_store_path, mode='train')
    create_meta(test_list, config.meta_store_path, mode='test')
    create_meta(val_lists, config.meta_store_path, mode='validation')
