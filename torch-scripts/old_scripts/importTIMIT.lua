require 'torch'
require 'hdf5'
local f = hdf5.open('timit_reduced_mfcc.h5', 'r')

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-load','111','Load specific parts of the dataset. [Training][Validation][Testing], example: 101 loads training and test sets. ')

opt = cmd:parse(arg)
opt.load=tonumber(opt.load)
print( opt.load)
--training set load
--if opt.load == ('100' or '101' or '110' or '111') then
  print ('Loading the training set...')
  training_targets=           f:read('/reduced/training/targets'):all()
  training_masks=             f:read('/reduced/training/masks'):all()
  training_labels_reduced=    f:read('/reduced/training/labels_reduced'):all()
  training_labels=            f:read('/reduced/training/labels'):all()
  training_feats=             f:read('/reduced/training/default'):all()
--end

--validation set load
--if opt.load == ('010' or '110' or '011' or '111') then
  print ('Loading the validation set...')
  validation_targets=           f:read('/reduced/validation/targets'):all()
  validation_masks=             f:read('/reduced/validation/masks'):all()
  validation_labels_reduced=    f:read('/reduced/validation/labels_reduced'):all()
  validation_labels=            f:read('/reduced/validation/labels'):all()
  validation_feats=             f:read('/reduced/validation/default'):all()
--end
--test set load

--if opt.load == (001 or '011' or '111' or '101') then  
  print ('Loading the test set...')
  test_targets=           f:read('/reduced/test/targets'):all()
  test_masks=             f:read('/reduced/test/masks'):all()
  test_labels_reduced=    f:read('/reduced/test/labels_reduced'):all()
  test_labels=            f:read('/reduced/test/labels'):all()
  test_feats=             f:read('/reduced/test/default'):all()
--end
f:close()
