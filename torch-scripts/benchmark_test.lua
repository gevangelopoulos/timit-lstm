require 'rnn'
require 'optim'
dofile('importTIMITtest.lua')
local logger=optim.Logger('test_results.log') --timestamp=1

local feats=test_feats:permute(2,1,3)
local labels=test_targets:permute(2,1,3)
labels=labels+1 --+1 to get class indices between 1-61.
local masks=test_masks:permute(2,1,3)


local inputSize = 39
local hiddenSize = 128
local outputSize = 61
local batchSize=1  --one sentence per time

local dsSize=feats:size(1) --size of the dataset. 192 for test, 400 for validation and 3696 for training



local maxSeqLength=feats:size(2) -- maxlength (in frames) for each sentence/example (619 for the test set, 777 for the training set and 742 for the validation set
local nClass = outputSize --how many classes


function build_data()
    local inputs={}
    local targets={}
    if use_saved then
      print('Using saved data...')
      inputs = torch.load('training.t7')
      targets = torch.load('targets.t7')
   else
    for i=1, dsSize do
      input=feats[i]:narrow(1,1,maxSeqLength)
      target=labels[i]:narrow(1,1,maxSeqLength)
      table.insert(inputs,input)
      table.insert(targets,target)
    end
   end
   return inputs, targets
end

local inputs, targets = build_data()
print('loading model...')
local rnn=torch.load('m.model')

local per_acc=0
logger:setNames{'Benchmarking on the test set'}
logger:setNames{'seqNum', 'PER'}
rnn:evaluate()
for seqNum=1,dsSize do
     
        local inputSequence = {}
        local expectedTarget = {}
        local max={}
        local index={}

        
        local mask=masks[seqNum]
        local seqLength=mask[mask:eq(1)]:size()[1]
        --print(seqLength)
        for j=1,seqLength do
        
         table.insert(inputSequence, inputs[seqNum][j])
         expectedTarget[j]=targets[seqNum][j] 
         --table.insert(expectedTarget, one_hot[j]+1)
        end
        
        rnn:zeroGradParameters()
        --rnn:clearState()
        start =torch.tic() 
        local out = rnn:forward(inputSequence)
        currT = torch.toc(start)
        print('Forward pass ' .. seqNum .. ' complete in: '.. currT .. 's')
        local predictedTarget={}
        for i=1,seqLength do
          max[i], predictedTarget[i] = torch.max(out[i],1)
        end
        local predictedTensor= torch.Tensor(seqLength)
        local expectedTensor=torch.Tensor(seqLength)
        local sum=0
        
        for i=1,seqLength do
          predictedTensor[i]=predictedTarget[i][1]
          expectedTensor[i]=expectedTarget[i][1]
         
          
          if predictedTensor[i]==expectedTensor[i] then
            sum=sum+1
          end
        end
        local per=(1-sum/(seqLength))*100
        per_acc=per_acc+per
        print('PER:' .. per)
        logger:add{seqNum, per}
               
  end
 
print('Average PER:' .. per_acc/dsSize)
logger:setNames{'Average PER', 'Epoch'}
logger:add{per_acc/dsSize, numEpochs}

