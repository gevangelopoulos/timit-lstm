require 'torch'
require 'hdf5'
local f = hdf5.open('timit_reduced_mfcc.h5', 'r')
  print ('Loading the training set...')
  training_targets=           f:read('/reduced/training/targets'):all()
  training_masks=             f:read('/reduced/training/masks'):all()
  training_labels_reduced=    f:read('/reduced/training/labels_reduced'):all()
  training_labels=            f:read('/reduced/training/labels'):all()
  training_feats=             f:read('/reduced/training/default'):all()
--end

f:close()
