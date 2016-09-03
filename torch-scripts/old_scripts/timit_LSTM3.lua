require 'rnn'
local use_saved = false
if not use_saved then
  dofile('importTIMITtest.lua')
  test_targets=test_targets:permute(2,1,3)
  test_feats=test_feats:permute(2,1,3)

  --training_targets=training_targets:permute(2,1,3)
  --training_feats=training_feats:permute(2,1,3)

end



local inputSize = 39


local hiddenSize = 128 --not used right now

local outputSize = 61

local dsSize=192 --size of timit_reduced/training

local batchSize=1 --one example per time

local seqLength=100 --length (in frames) for each sentence/example
local nClass = 61

function save(inputs, targets, rnn)
   torch.save('training.t7', inputs)
   torch.save('targets.t7', targets)
   torch.save('trained-model.t7', rnn)
end


function build_data()
    local inputs={}
    local targets={}
    if use_saved then
      print('Using saved data...')
      inputs = torch.load('training.t7')
      targets = torch.load('targets.t7')
   else
    for i=1, dsSize do
      local input=test_feats[i]:narrow(1,1,seqLength)
      local target=test_targets[i]:narrow(1,1,seqLength)
      --local input=training_feats[i]:narrow(1,1,seqLength)
      --local target=training_targets[i]:narrow(1,1,seqLength)
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
      rnn = nn.Sequential() 
      :add(nn.FastLSTM(inputSize, hiddenSize))
      :add(nn.Linear(hiddenSize,outputSize))
      :add(nn.LogSoftMax())
      
      rnn=nn.Sequencer(rnn)
  end
   return rnn
end

-- MAIN
--two tables to hold the *full* dataset input and target tensors
inputs, targets = build_data()
rnn = build_network(inputSize, hiddenSize, outputSize)

crit=nn.ClassNLLCriterion()
seqC = nn.SequencerCriterion(crit)

rnn:training()

for numEpochs=0,20 do
  print('Epoch ',numEpochs)
  local lr
  local err = 0
  local err_sum=0
  local start = torch.tic()
  if numEpochs<3 then
    lr=0.05
  elseif numEpochs<5 then
    lr=0.005
  elseif numEpochs<10 then
    lr=0.0005
  elseif numEpochs<20 then
    lr=0.00005
  end
  print('Learning Rate: ',lr)
   for seqNum=1,dsSize do
     
        local inputSequence = {}
         expectedTarget = {}

        local i=targets[seqNum]:long()+1
        local one_hot=torch.zeros(seqLength,outputSize)
        one_hot:scatter(2,i,1)
        
      
        for j=1,seqLength do
        
         table.insert(inputSequence, inputs[seqNum][j])
         table.insert(expectedTarget, one_hot[j]+1)
        end
        
        rnn:zeroGradParameters()
         out = rnn:forward(inputSequence)
        --print('The output is',out)
        
        
        err = seqC:forward(out, expectedTarget)
  
        err_sum=err_sum+err
       
        
        
        gradOut = seqC:backward(out, expectedTarget)
        --print('gradout is ', gradOut)
        rnn:backward(inputSequence, gradOut)
        --We update params at the end of each batch
        
        rnn:updateParameters(lr)
        
        print ('Epoch #', numEpochs .. 'Utterance #', seqNum .. ' of ', dsSize .. '. Loss: ', err)
        
        
        
  end
    local currT=torch.toc(start)
  print('Epoch #', numEpochs .. ' average loss', err_sum/dsSize .. 'in ', currT .. ' s')
  print('Saving inputs, targets, model...')
  save(inputs, targets, rnn)
end



--[[
for numEpochs=0,10,1 do
  local err =0;
  
  for each sequence (sentence of fixed length lets say 700) do
    pass the whole sequence into the network
      out=rnn:forward(inputSequence)
      pass the expected output to the criterion, this should be either 700x1 or 700x61 one-hot 
      err = err + seqC:forward(out, expectedTarget)
      gradOut=seqC:backward(out, expectedTarget)
      rnn:backward(inputSequence,gradOut)
      rnn:updateParameters(0.05)
      rnn:zeroGradParameters()
      print(err)
  end
--]]





