import scipy.io
import os

our_vgg = scipy.io.loadmat(os.path.join('C:\\','Users','Geo','Dropbox','Project A','vot_new','VGG_TRAIN', 'imagenet-vgg-verydeep-19'))
orig_vgg = scipy.io.loadmat(os.path.join('C:\\','Users','Geo','Desktop','Study', 'Semester 6','project A','code', 'VGG_TRAIN', 'imagenet-vgg-verydeep-19'))

print(scipy.equal(our_vgg, orig_vgg))