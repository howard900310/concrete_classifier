import os
# filenames = os.listdir(os.getcwd())

def change_file_name(orig_label, new_label, image_folder=str):
    '''
    change the image's label from orig_label to new_label.

    orig_label = 3

    new_label = 1
    
    output : 000007_256_256_3.jpg --> 000007_256_256_1.jpg
    '''

    # find the image's location
    filenames = os.listdir(os.getcwd() + image_folder) 

    data_path = os.getcwd() + image_folder + '\\'

    for name in filenames:
        name_split = name.split(sep = '_')
        # print(name_split)
        if name_split[-1] == str(orig_label) + '.jpg':  # if label is orig_label -> change label
            orig_name = data_path + name
            new_name = data_path + name_split[0]+ '_' + name_split[1] + '_' + name_split[2] + '_' + str(new_label) + '.jpg'
            os.rename(orig_name, new_name)

if __name__ == '__main__':
    change_file_name(orig_label = 1, new_label = 3, image_folder = '\\output_test\\1_spalling')


