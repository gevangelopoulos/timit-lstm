require 'rnn'
inputs = torch.load('training.t7')
targets = torch.load('targets.t7')
rnn = torch.load('trained-model.t7')
print(rnn)

os.execute('rm -rf training.t7 targets.t7 trained-model.t7')
torch.save('training.t7', inputs)
torch.save('targets.t7', targets)
torch.save('trained-model.t7', rnn)
