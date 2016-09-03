require 'torch'
require 'hdf5'
local f = hdf5.open('timit_reduced_mfcc.h5', 'r')
 
  print ('Loading the test set...')
  test_targets=           f:read('/reduced/test/targets'):all()
  test_masks=             f:read('/reduced/test/masks'):all()
  test_labels_reduced=    f:read('/reduced/test/labels_reduced'):all()
  test_labels=            f:read('/reduced/test/labels'):all()
  test_feats=             f:read('/reduced/test/default'):all()
--end
f:close()
