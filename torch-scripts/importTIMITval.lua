require 'torch'
require 'hdf5'
local f = hdf5.open('timit_reduced_mfcc.h5', 'r')

  print ('Loading the validation set...')
  validation_targets=           f:read('/reduced/validation/targets'):all()
  validation_masks=             f:read('/reduced/validation/masks'):all()
  validation_labels_reduced=    f:read('/reduced/validation/labels_reduced'):all()
  validation_labels=            f:read('/reduced/validation/labels'):all()
  validation_feats=             f:read('/reduced/validation/default'):all()
--end

f:close()
