require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size)
  local inputs={}
  table.insert(inputs,nn.Identity()())
  table.insert(inputs,nn.Identity()())
  table.insert(inputs,nn.Identity()())
  local input=inputs[1]
  local prev_c=inputs[2]
  local prev_h=inputs[3]

  local i2h=nn.Linear(input_size, 4*rnn_size)(input)
  local h2h=nn.Linear(rnn_size,4*rnn_size)(prev_h)
  local preactivations=nn.CAddTable()({i2h,h2h})

  local pre_sigmoid_chunk=nn.Narrow(2,1,3*rnn_size)(preactivations)
  local all_gates=nn.Sigmoid()(pre_sigmoid_chunk)

  local in_chunk=nn.Narrow(2,3*rnn_size+1,rnn_size)(preactivations)
  local in_transform=nn.Tanh()(in_chunk)

  local in_gate=nn.Narrow(2,1,rnn_size)(all_gates)
  local forget_gate=nn.Narrow(2,rnn_size+1,rnn_size)(all_gates)
  local out_gate=nn.Narrow(2,2*rnn_size+1,rnn_size)(all_gates)

  local c_forget=nn.CMulTable()({forget_gate,prev_c})
  local c_input=nn.CMulTable()({in_gate,in_transform})

  local next_c=nn.CAddTable()({c_forget,c_input})

  local c_transform=nn.Tanh()(next_c)
  local next_h=nn.CMulTable()({out_gate, c_transform})

  outputs={}
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  return nn.gModule(inputs, outputs)
end

return LSTM
