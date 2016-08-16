from __future__ import print_function

import joblib
import os
import wand.image
import wand.exceptions
import sys


def get_filename_list(image_dir):
    filename_list = list()
    for class_name in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, class_name)
        if class_name.startswith('.') or not os.path.isdir(class_dir):
            continue
        for (file_dir, _, file_names) in os.walk(class_dir):
            for file_name in file_names:
                if not file_name.endswith('.jpg'):
                    continue
                filename_list.append(os.path.join(file_dir, file_name))
    return filename_list

def load_file(num_file, filename):
    try:
        image = wand.image.Image(filename=filename)
    except wand.exceptions.CorruptImageError:
        os.remove(filename)
        print('Remove %s' % file_path)
    print('\033[2K\r# %d' % num_file, end='')
    sys.stdout.flush()

parallel = joblib.Parallel(n_jobs=8, backend='threading')
parallel(joblib.delayed(load_file)(num_file, filename) for (num_file, filename) in enumerate(get_filename_list('/mnt/data/dish-clean')))
