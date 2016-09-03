require 'rnn'
require 'optim'
local date=os.date('%Y_%m_%d_%X')
--local exp_path='./experiments'
local exp_id = string.format('%s_%s', 'experiment', date)
local log_id= exp_id .. '.log'
logger=optim.Logger(log_id)
local use_saved = false
if not use_saved then
  --[[
  dofile('importTIMITtest.lua')
  feats=test_feats:permute(2,1,3)
  labels=test_targets:permute(2,1,3)
  labels=labels+1
  masks=test_masks:permute(2,1,3)
 --]]
  
  dofile('importTIMITtrain.lua')
  feats=training_feats:permute(2,1,3)
  labels=training_targets:permute(2,1,3)
  labels=labels+1 -- +1 because the targets in the hdf5 file are labeled 0-60 while we want them to be 1-61.
  masks=training_masks:permute(2,1,3)
  
  
end


local inputSize = 39
local hiddenSize = 128 
local outputSize = 61
local batchSize=1 --one example per time

local dsSize=feats:size(1) --size of timit_reduced/training
local seqLength=feats:size(2) --length (in frames) for each sentence/example
local nClass = outputSize


function save(model, exp_id, numEpochs)
   local filename = exp_id .. '.model.' .. numEpochs
   print('Saving model as ' .. filename)
   torch.save(filename, model)
end


function build_data()
    local inputs={}
    local targets={}
    if use_saved then
      print('Using saved data...')
      inputs = torch.load('inputs.t7')
      targets = torch.load('targets.t7')
   else
    for i=1, dsSize do
      local input=feats[i]:narrow(1,1,seqLength)
      local target=labels[i]:narrow(1,1,seqLength)
      
      table.insert(inputs,input)
      table.insert(targets,target)
    end
   end
   return inputs, targets
end

function build_network(inputSize, hiddenSize, outputSize)
  if use_saved then
      print('Using saved model...')
      rnn = torch.load('trained-model.t7')
  else
  --BLSTM
      --require 'nngraph'
      --nn.FastLSTM.usenngraph=true
      fwd = nn.FastLSTM(inputSize, hiddenSize)
      blstm=nn.BiSequencer(fwd) --BiSequencer creates a clone of fwd to be trained backwards
      rnn = nn.Sequential() 
      :add(blstm)
      :add(nn.Sequencer(nn.Linear(hiddenSize*2,outputSize))) -- hiddenSize*2 due to BLSTM
   
   --[[
   --GRU
       rnn = nn.Sequential() 
       :add(nn.GRU(inputSize, hiddenSize))
       :add(nn.Linear(hiddenSize,outputSize))
       rnn=nn.Sequencer(rnn)
   --]]
   end
   return rnn
end

-- MAIN
--two tables to hold the *full* dataset input and target tensors
inputs, targets = build_data()
rnn = build_network(inputSize, hiddenSize, outputSize)
logger:add{['model architecture'] = tostring(rnn)}

crit=nn.CrossEntropyCriterion()
seqC = nn.SequencerCriterion(crit)


parameters, gradParameters = rnn:getParameters()
light_rnn=rnn:clone('weight','bias','running_mean','running_std')

local lr=0.005
rnn:training()

for numEpochs=0,100 do
logger:setNames{'Epoch', 'Sequence', 'Loss'}
  print('Epoch ',numEpochs)
  
  local err = 0
  local err_sum=0
  local start = torch.tic()
  
  print('Learning Rate: ',lr)
  --logger:add{['learning_rate']=lr}
   for seqNum=1,dsSize do
     
        inputSequence = {}
        expectedTarget = {}

        --find out the sequence length of each sequence
        mask=masks[seqNum]
        seqLength=mask[mask:eq(1)]:size()[1]
        
        for j=1,seqLength do
        
         table.insert(inputSequence, inputs[seqNum][j])
         expectedTarget[j]=targets[seqNum][j] 
        
        end
        --return
        --print('Utterance #', seqNum .. 'of length ' .. seqLength)
        rnn:zeroGradParameters()
         out = rnn:forward(inputSequence)
      
        err = seqC:forward(out, expectedTarget)
        err_sum=err_sum+err     
        
        gradOut = seqC:backward(out, expectedTarget)
        --print('gradout is ', gradOut)
        rnn:backward(inputSequence, gradOut)
        --We update params at the end of each batch
        
        rnn:updateParameters(lr)
        
        print ('Epoch #', numEpochs .. 'Utterance #', seqNum .. ' of ', dsSize .. '. Loss: ', err)
        logger:add{numEpochs,seqNum,err}
        
        
        --]]
  end
  
    local currT=torch.toc(start)
    average_loss=err_sum/dsSize
  print('Epoch #', numEpochs .. ' average loss', average_loss .. 'in ', currT .. ' s')
  logger:setNames{'Epoch', 'average_loss', 'time', 'learning_rate'}
  logger:add{numEpochs, average_loss, currT,lr}
  
  --Checkpoint
  local checkpoint = {
    exp_id=exp_id,
    numEpochs=numEpochs,
    average_loss=average_loss
   }
    
  save(light_rnn,exp_id,numEpochs)
  counter=numEpochs --weird bug that numEpochs is out of scope in benchmark_validation.lua
  print('Model is now tested on the validation set...')
  dofile('benchmark_validation.lua')
  if numEpochs!=0 then
    if average_loss>=previous_average_loss then
      lr=lr/2
    else
      lr=lr*1.01
    end
  else
    previous_average_loss=average_loss
  end
end





