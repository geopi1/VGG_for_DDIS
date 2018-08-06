import numpy as np
import os
import scipy.misc
import config
import image_crop
import tensorflow as tf



def TestResult(whole_image_path, true_crop_path, resize=True, fliplr=False):
    
    # save true_crop as np array
    TruthPic = np.float32(scipy.misc.imread(true_crop_path))
    # find the h,w of true_crop
    TruthSize = TruthPic.shape

    # load the full frame
    BigPic = np.float32(scipy.misc.imread(whole_image_path))
    # find the shape of the frame (used to determine the boarders of the crops)
    BigSize = BigPic.shape

    # Resize the true_crop of the image so it fits the input-size of the net
    if resize:
        TruthPic = scipy.misc.imresize(TruthPic, size=config.TRAIN.resize, interp='bilinear', mode=None)
        BigPic = scipy.misc.imresize(TruthPic, size=config.TRAIN.resize, interp='bilinear', mode=None)
    if fliplr:
        TruthPic = np.fliplr(TruthPic)

    min_loss = 10**100
    best_crop_loc = [0, 0]


    # Matrix to hold losses of each relevant crop
    test_g_loss = np.zeros((BigSize[0] - TruthSize[0], BigSize[1] - TruthSize[1]), dtype=float) #Since we want entire crop to be inside orig image, matrix is smaller
    for row in range(0,BigSize[0] - TruthSize[0]):
        for col in range(0,BigSize[1] - TruthSize[1]):
            # print(str(row)+" "+str(col))
            CurCrop = BigPic[row:row+TruthSize[0], col:col + TruthSize[1], :]
            # print(CurCrop.shape)
            if resize:
                CurCrop = scipy.misc.imresize(CurCrop, size=config.TRAIN.resize, interp='bilinear', mode=None)
            if fliplr:
                CurCrop = np.fliplr(CurCrop)
            CurCrop = np.expand_dims(CurCrop, axis=0)
            # We set overlap_input such that the loss for curCrop will be the actual loss
            # and not distance from our expected loss, as it is while training.
            output = sess.run(CX_content_loss, feed_dict={input_A: CurCrop, input_B: TruthPic, overlap_input: 1 - config.TRAIN.epsilon})
            test_g_loss[row][col] = eval(CX_content_loss)

    # Returns all minimum locations, by ([rowMin1,rowMin2,...],[colMin1,colMin2,...])
    TestBestMatches=np.where(test_g_loss == test_g_loss.min())
    for ind in range(0, len(TestBestMatches[0])):
        UpperLeftLoc = (TestBestMatches[0][ind], TestBestMatches[1][ind])
        BestMatch = BigPic[UpperLeftLoc[0]: UpperLeftLoc[0] + TruthSize[0], UpperLeftLoc[1]: UpperLeftLoc[1] + TruthSize[1], :]
        scipy.misc.imsave(os.path.join(ResultsDir, "BestMatch" + str(ind) + ".bmp"), BestMatch)


    # calculate how close the min crop is to the TruthPic (how much overlap)
    # get the location of the true crop
    truth_loc = image_crop.get_bounding_box()

    # calculate the overlap
    overlap = 100 * (TruthSize[1] - abs(best_crop_loc[1] - int(truth_loc[1]))) * (TruthSize[0] -
                        (abs(truth_loc[0] - best_crop_loc[0]))) / (TruthSize[1] * TruthSize[0])
