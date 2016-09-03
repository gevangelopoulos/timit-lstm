require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size)
  local inputs={}
  table.insert(inputs,nn.Identity()())
  table.insert(inputs,nn.Identity()())
  table.insert(inputs,nn.Identity()())
  table.insert(inputs,nn.Identity()())
  local input=inputs[1]
  local prev_c=inputs[2]
  local prev_h=inputs[3]
  local coupling=inputs[4]
  
  -- Linear Transformations. Input is input, performs Wxi*input, Wxo*input, Wxc*input.
  local i2h=nn.Linear(input_size, 3*rnn_size)(input)
  
  -- Linear Transformations. Input is prev_h, performs Whi*prev_h, Who*prev_h, Whc*prev_h.
  local h2h=nn.Linear(rnn_size,3*rnn_size)(prev_h)
  
  -- Addition. Performs Wxi*input+Whi*prev_h, ...
  local preactivations=nn.CAddTable()({i2h,h2h})

 -- Slices the preactivations table, picking two parts representing the input and output gates before the activations. The forget gate will be calculated later.
  local pre_i=nn.Narrow(2,1,rnn_size)(preactivations)
  local pre_o=nn.Narrow(2,rnn_size+1,rnn_size)(preactivations)
  
  --Applies activation functions to the gates. The forget gate is calculated by subtracting the in_gate from the "coupling" which will be 1.
  local in_gate=nn.Sigmoid()(pre_i)
  local forget_gate=nn.CSubTable()({coupling,in_gate}) 
  local out_gate=nn.Sigmoid()(pre_o)
  
  -- Slices the preactivations table, picking the part that is the state update.
  local in_chunk=nn.Narrow(2,2*rnn_size+1,rnn_size)(preactivations)
  --Aplies Tanh to the state update.
  local in_transform=nn.Tanh()(in_chunk)
  
  
  -- Elementwise calculation of the parts that the new state should "forget" and "remember".
  local c_forget=nn.CMulTable()({forget_gate,prev_c})
  local c_input=nn.CMulTable()({in_gate,in_transform})
  -- Creation of the new state by adding the two previous parts.
  local next_c=nn.CAddTable()({c_forget,c_input})

  -- Applies Tanh activation on the internal state.
  local c_transform=nn.Tanh()(next_c)
  -- Elementwise calculation of the next output.
  local next_h=nn.CMulTable()({out_gate, c_transform})

  
  outputs={}
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  return nn.gModule(inputs, outputs)
end

return LSTM
