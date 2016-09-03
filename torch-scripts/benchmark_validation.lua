require 'rnn'
require 'optim'
local matio = require 'matio'
dofile('importTIMITval.lua')
--dofile('examine_model.lua')
local logger=optim.Logger('val_results.log',1) --timestamp=1

local feats=validation_feats:permute(2,1,3)
local labels=validation_targets:permute(2,1,3)
labels=labels+1 --+1 to get class indices between 1-61.
local masks=validation_masks:permute(2,1,3)

matio.save('validation.mat',{feats=feats, labels=labels})

local dsSize=feats:size(1) --size of the dataset. 192 for test, 400 for validation and 3696 for training
local maxSeqLength=feats:size(2) -- maxlength (in frames) for each sentence/example (619 for the test set, 777 for the training set and 742 for the validation set



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
--print('loading model...')
local rnn=torch.load('m.model')
--print(rnn)
local class_index={}

for i=1,61 do class_index[i]=i end
conf = optim.ConfusionMatrix( class_index )
conf:zero()
local per_acc=0
logger:setNames{'Benchmarking on the validation set'}
logger:setNames{'seqNum', 'PER'}

seqLengths={}
rnn:evaluate()
for seqNum=1,dsSize do
     
        local inputSequence = {}
        local expectedTarget = {}
        local max={}
        local index={}
        
        local mask=masks[seqNum]
        local seqLength=mask[mask:eq(1)]:size()[1]
        seqLengths[seqNum]=seqLength
        for j=1,seqLength do
        
         table.insert(inputSequence, inputs[seqNum][j])
         expectedTarget[j]=targets[seqNum][j] 
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
          conf:add(predictedTarget[i][1], expectedTarget[i][1])
          
          if predictedTensor[i]==expectedTensor[i] then
            sum=sum+1
          end
        end
        local per=(1-sum/(seqLength))*100
        per_acc=per_acc+per
        print('PER:' .. per)
        logger:add{seqNum, per}
               
  end
  average_per=per_acc/dsSize
print('Average PER:' .. average_per)
logger:setNames{'Epoch', 'Average_PER'}
logger:add{counter, average_per}
logger:setNames{'============'}

