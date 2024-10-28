# CS5487 demo script for Programming Assignment 2
#
# The script has been tested with python 2.7.6
#
# It requires the following modules:
#   numpy 1.8.1
#   matplotlib v1.3.1
#   scipy 0.14.0
#   Image (python image library)

import pa2
import numpy as np
import pylab as pl
from PIL import Image
import scipy.io as sio

def demo():
    import scipy.cluster.vq as vq

    image_name = '21077'
    methods=['Kmeans', 'EmGmm', 'Meanshift']

    for method in methods:
        file = open(f'/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/MLPA/output/image_segmentation/labels/{image_name}-{method}.txt', 'r')
        labels = np.array(list(filter(None, file.readline().split(' '))))

        ## load and show image
        img = Image.open(f'../images/{image_name}.jpg')
        # pl.subplot(1,3,1)
        # pl.imshow(img)

        ## extract features from image (step size = 7)
        X,L = pa2.getfeatures(img, 7)

        segm = pa2.labels2seg(labels,L)
        pl.subplot(1,2,1)
        pl.imshow(segm)

        # color the segmentation image
        csegm = pa2.colorsegms(segm, img)
        pl.subplot(1,2,2)
        pl.imshow(csegm)
        pl.savefig(f'../output/image_segmentation/results/{image_name}-{method}.png', format='png', dpi=300)
        # pl.show()

def main():
    demo()
if __name__ == '__main__':
    main()
