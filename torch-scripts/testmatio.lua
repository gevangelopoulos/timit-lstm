require 'torch'
local matio = require 'matio'

data=torch.rand(5,5)
matio.save('test1.mat', data)

data1=torch.rand(5,5)
data2=torch.rand(2,3):float()
matio.save('test2.mat',{t1=data,t2=data2})

data1=torch.rand(2,3):float()
matio.save('test3.mat',{t1=data,t2='helloo', t3=9, t4=false})

