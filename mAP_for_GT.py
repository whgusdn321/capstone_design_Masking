from __future__ import absolute_import
from __future__ import division
import os
import shutil
import logging
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
#from ..base import VisionDataset

'''
PATH_TO_3 은 은기가 annotation해놓은 폴더(jpg, xml동시에 들어가 있는 폴더)의 path
gt_path 는 gt_path txt를 만들 dir ->미리 생성되 있어야 함
image_path 는 이미지가 들어있을 dir ->미리 생성되 있어야 함
'''


PATH_TO_3 = r'C:\Users\suer0426\Desktop\testset\3조\MX9 1080HD in car black box video'

gt_path = r'C:\Users\suer0426\Desktop\testset\mAP_gt'
image_path = r'C:\Users\suer0426\Desktop\testset\mAP_images'

for(path, dir, filenames) in os.walk(PATH_TO_3):
    for i, filename in enumerate(filenames):
        print('filename :', filename)
        if filename[-3:] in 'jpg':
            full_image_path = os.path.join(path, filename)
            shutil.copy(full_image_path, image_path)
            #copy to the images file
        if filename[-3:] in 'xml':
            full_anot_path = os.path.join(path, filename)
            root = ET.parse(full_anot_path).getroot()
            label = []
            for obj in root.iter('object'):
                print('is called ! full_anot_path is {}'.format(full_anot_path))
                difficult = int(obj.find('difficult').text)
                cls_name = obj.find('name').text.strip().lower()
                xml_box = obj.find('bndbox')
                xmin = int((float(xml_box.find('xmin').text) - 1))
                ymin = int((float(xml_box.find('ymin').text) - 1))
                xmax = int((float(xml_box.find('xmax').text) - 1))
                ymax = int((float(xml_box.find('ymax').text) - 1))
                label.append([cls_name, xmin, ymin, xmax, ymax, difficult])
            print('len_label is :',len(label))
            if not len(label) == 0:
                file2 = open('{}/{}.txt'.format(gt_path, filename[:-4]), "w")
                print('{}/{}.txt'.format(gt_path, filename[:-4]))
                for (cls_name, xmin, ymin, xmax, ymax, difficult) in label:
                    str_to_write = cls_name+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)
                    print('str_to_write : ', str_to_write)
                    file2.write(str_to_write+'\n')

            else:
                file3 = open('{}/{}.txt'.format(gt_path, 'violated_file'), "a")
                file3.write('Warning : There is no 0 boxes in file : {}\n'.format(full_anot_path))
                print('Warning : There is no 0 boxes in file : {}'.format(full_anot_path))
                print('And this file is not in Annotation, and text(val.txt or train.txt)')





    #file_path = os.path.join(path, filename)
    #y_class = int(file_path[-5])
    #if y_class != 0:
    #    y_class = 1
    #x_whole.append(file_path)
    #y_whole.append(y_class)