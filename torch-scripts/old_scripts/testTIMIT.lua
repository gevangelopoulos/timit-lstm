require 'rnn'
dofile('importTIMITtest.lua')
--dofile('examine_model.lua')

test_feats=test_feats:permute(2,1,3)
test_targets=test_targets:permute(2,1,3)
test_targets=test_targets+1 --+1 to get class indices between 1-61.
local inputSize = 39


local hiddenSize = 128 --not used right now

local outputSize = 61
local dsSize=192 --size of timit_reduced/training

local batchSize=1 --one example per time

local seqLength=619 --length (in frames) for each sentence/example
local nClass = 61


function build_data()
    local inputs={}
    local targets={}
    if use_saved then
      print('Using saved data...')
      inputs = torch.load('training.t7')
      targets = torch.load('targets.t7')
   else
    for i=1, dsSize do
      input=test_feats[i]:narrow(1,1,seqLength)
      target=test_targets[i]:narrow(1,1,seqLength)
      table.insert(inputs,input)
      table.insert(targets,target)
    end
   end
   return inputs, targets
end

inputs, targets = build_data()
per_acc=0
for seqNum=1,10 do
     
        inputSequence = {}
        expectedTarget = {}
        max={}
        index={}
        --i=targets[seqNum]:long()+1
        --one_hot=torch.zeros(seqLength,outputSize)
        --one_hot:scatter(2,i,1)
        
      
        for j=1,seqLength do
        
         table.insert(inputSequence, inputs[seqNum][j])
         expectedTarget[j]=targets[seqNum][j] 
         --table.insert(expectedTarget, one_hot[j]+1)
        end
        
        rnn:zeroGradParameters()
        --rnn:clearState()
        
        start =torch.tic() 
        out = rnn:forward(inputSequence)
        currT = torch.toc(start)
        print('Forward pass complete in: '.. currT .. 's')
        predictedTarget={}
        for i=1,seqLength do
          max[i], predictedTarget[i] = torch.max(out[i],1)
        end
        predictedTensor= torch.Tensor(seqLength)
        expectedTensor=torch.Tensor(seqLength)
        sum=0
        silCount=0
        
        for i=1,seqLength do
          predictedTensor[i]=predictedTarget[i][1]
          expectedTensor[i]=expectedTarget[i][1]
          if predictedTensor[i]==28 then silCount=silCount+1 end
          
          if predictedTensor[i]==expectedTensor[i] and (predictedTensor[i]~=28) then
            sum=sum+1
          end
        end
        per=(1-sum/(seqLength-silCount))*100
        per_acc=per_acc+per
        print('PER:' .. per)
        
       
        --x=torch.cat(expectedTensor,predictedTensor,2)
        --print('Tensor x is a concatenation of expectedTensor and predictedTensor')
               
  end
 
print('PER:' .. per_acc/10)

