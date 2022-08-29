import os
# filenames = os.listdir(os.getcwd())

def change_file_name(orig_label, new_label, image_folder=str):
    '''
    orig_label to new_label
    '''
    filenames = os.listdir(os.getcwd() + image_folder)

    x = os.getcwd() + image_folder + '\\'
    print(x)
    for name in filenames:
        name_split = name.split(sep = '_')
        print(name_split)
        if name_split[-1] == str(orig_label) + '.jpg':
            os.rename( x + name, x + name_split[0]+ '_' + name_split[1] + '_' + name_split[2] + '_' + str(new_label) + '.jpg')

if __name__ == '__main__':
    change_file_name(orig_label = 0, new_label = 1, image_folder = '\\0_23_cleaned0829\\1_spalling')

