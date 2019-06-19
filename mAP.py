from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os
import mxnet as mx
import numpy as np
from demo_faster_rcnn import *

"""
To calculate map, we have to make two folders (optionally, 3 folders if we want animation while calculating mAP.

First, there is mAP_images which contains all the images we want to include to calculate mAP
Second, there is mAP_detection which contains all the detected scores, bbox coordinates.
Last, there is maP_gt which contains all ground truth 
  
"""


image_PATH = r'C:\Users\suer0426\Desktop\Desktop (2)\mAP_images'
detecion_PATH = r"C:\Users\suer0426\Desktop\Desktop (2)\mAP_detection"

filenames = os.listdir(image_PATH)
im_fname = []
full_fname = []

for filename in filenames:
    print('filename : ',filename)
    full_filename = os.path.join(image_PATH, filename)
    partial_filename = filename[:-4]
    im_fname.append(partial_filename)
    full_fname.append(full_filename)
    print('full_filename : ',full_filename)
    print('partial_filename =', partial_filename)

##anyway. im_fname will be like
# im_fname = ['Normal_Videos_913_x264_90',
# 'Normal_Videos_913_x264_350',
# 'Normal_Videos_913_x264_310',
# 'Normal_Videos_913_x264_140',
# 'Normal_Videos_913_x264_100']

# for(path, dir, filenames) in os.walk(image_PATH):
#     for filename in filenames:
#         image_path = os.path.join(path, filename)
#         im_fname.append(image_path)

# for i in range(len(im_fname)):
#     im_fname[i] = os.path.join(image_PATH, im_fname[i])+'.jpg'
#print('imfname is ', im_fname)


x, orig_img, real_orig = data.transforms.presets.rcnn.load_test(full_fname)

print('processed done!')

print('len(orig_img)', len(orig_img))
print('len(real_orig)', len(real_orig))

print('orig_img[0].shape',orig_img[0].shape)
print('real_orig[0].shape',real_orig[0].shape)
print('full_fname[0]: ',full_fname[0])




for i in range(len(full_fname)):
    box_ids, scores, bboxes = net2(x[i].copyto(ctx[1]))
    #box_ids2, scores2, bboxes2 = net2(x[i].copyto(ctx[1]))

    valid_scores = np.where(scores.asnumpy() >= 0.5)[1] # shape (n,) nparray
    face_index = np.where(box_ids.asnumpy() == 0)[1] # shape (m,) nparray

    # valid_scores2 = np.where(scores2.asnumpy() >= 0.5)[1]
    # car_index = np.where(box_ids2.asnumpy() == 0)[1]

    print('person_index : ',face_index)
    if np.intersect1d(valid_scores, face_index).shape[0] != 0:
        mask_index = np.intersect1d(valid_scores, face_index)
    else:
        mask_index = np.asarray([])

    print('mask_index : ',mask_index)


    h_ratio = real_orig[i].shape[0]/orig_img[i].shape[0]
    w_ratio = real_orig[i].shape[1]/orig_img[i].shape[1]

    if mask_index.shape[0] != 0:

        selected_scores = scores[0, mask_index, 0] # like [0.97, 0.98, 0.01, 0.5]
        sorted_index = np.argsort(-1*selected_scores.asnumpy())
        #time.sleep(1)

        f = open('{}/{}.txt'.format(detecion_PATH, im_fname[i]), "w")
        for index in sorted_index:
            real_index = mask_index[index]
            # if real_index in person_index.tolist():
            #     strr = 'person'
            # else:
            #     strr = 'car'
            strr = 'car'
            score = scores[0, real_index, 0].asscalar()
            xmin, ymin, xmax, ymax = bboxes[0, real_index]
            xmin = int(xmin.asscalar()*w_ratio)
            ymin = int(ymin.asscalar()*h_ratio)
            xmax = int(xmax.asscalar()*w_ratio)
            ymax = int(ymax.asscalar()*h_ratio)
            write_string = strr + ' ' + str(score)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)
            print('write_string : ', write_string)
            f.write(write_string+'\n')

        '''
        TODO: open the file with name im_fname[i].txt, like asdfq.txt in some folder,
        write it. 
        for index in np.sort(np.concatenate(person_index, car_index, axis=-1)):
            if index is in person_index:
                object = person
                score = scores[0, index] # should be scalar, not ndarray
                xmin = int(@), ymin = int(@), xmax = int(@) ymax = int(@)
                f1.write(objec, score, xmin, ymin, xmax, ymax)
        '''
        # f = open('{}/{}.txt'.format(detection_path, im_fname[i]), "w")
        #    break
        #for index in mask_index:

    else :
        f = open('{}/{}.txt'.format(detecion_PATH, im_fname[i]), "w")
        '''
                TODO: open the file with name im_fname[i].txt, like asdfq.txt in some folder,
                write it. 
                just make file, and don't write anything!
                '''

        #plt.show()