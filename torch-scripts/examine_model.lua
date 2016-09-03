require 'rnn'
local matio = require 'matio'

print('loading model...')
rnn = torch.load('m.model')

print(rnn)

w,dw=rnn:parameters()

matio.save('parameters.mat',{w1=w[1],w2=w[2], w3=w[3],w4=w[4],w5=w[5],w6=w[6],w7=w[7],w8=w[8]})

