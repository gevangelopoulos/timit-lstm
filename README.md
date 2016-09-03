# Long Short-Term Memory Neural Networks for Automatic Speech Recogniotion on the TIMIT dataset

This repo contains [Torch] scripts and models I created when working on my master's thesis [Efficient Hardware Mapping of LSTM Neural Networks for Speech Recognition], in the [ESAT-MICAS] lab of KU Leuven, Belgium, from February till July 2016.

The code here is by no means perfect; It's a collection of patchy scripts aimed to do their job correctly and fast.
The collections includes:

  - Scripts that import the TIMIT Acoustic Dataset from an HDF5 file format into Torch tensors
  - Scripts that set up a Bidirectional LSTM architecture using modules from torch-rnn
  - Scripts that use TIMIT's training set to train the LSTM architecture using backpropagation through time
  - Scripts that evaluate the performance of the trained models on TIMIT's validation and test set.

Some more functionality:
  - Logging scripts that report the loss function during training and the framewise classification error during testing
  - Scripts that export a timestamped snapshot of the model every epoch.

Pretrained models with snapshots over epochs are also included, since an LSTM parameter analysis over training could be useful. The included models were trained for a large number of epochs but only the first 40 epochs are included here, to save space. In any case, almost all models reached their peak performance of about 70% in accuracy (30% Framewise Classification Error) in less than 10-15 epochs, so if you want to use a model in your research, check the logs to see which snapshot you should grab.

Having a basic understanding of the Lua programming language and the Torch framework can be advantageous, and tweaking the code to use different LSTM sizes or architectures shouldn't be too difficult. Knowing a little more about machine learning could help you expand this model into a larger, customized application such as a deep LSTM, a convnet-LSTM combination, a GRU and more.

### Dependencies


This project uses parts of many other projects. While the dependencies are not well defined, the following list should give you pretty much anything you'll need to reproduce these results.

* [The TIMIT Dataset] - The TIMIT dataset used for this research, which was licensed from the Linguistic Data Consortium.
* [sph2pipe] - A tool that can convert sphere files from the LDC corpus into wav files.
* [Torch] - The Torch machine learning framework.
* [rnn] - Efficient reusable RNNs and LSTMs for Torch
* [h5py] - An HDF5 python library
* [torch-hdf5] - A Torch HDF5 library
* [HTK MFCC MATLAB] - A matlab library for calculating MFCC coefficients in matlab.


### More interesting, useful things

* [char-rnn] - Useful and fun character-level generative language models in Torch. Give it a text in a style, and it will learn to generate similar texts!
* [Andrej Karpathy's blog] - A cool blog from a cool guy working on deeplearning.
* [Christopher Olah's blog] - Another cool blog talking about deep learning, RNNs, LSTMs, data visualization and more.
* [LDC] - How to use data from the Linguistic Data Consortium
* [matio] - A .mat file i/o library.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [The TIMIT Dataset]: <https://catalog.ldc.upenn.edu/ldc93s1>
   [sph2pipe]:<https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools>
   [Torch]: <http://torch.ch/>
   [rnn]: https://github.com/Element-Research/rnn>
   [h5py]: <https://github.com/h5py/h5py>
   [torch-hdf5]: <https://github.com/deepmind/torch-hdf5>
   [HTK MFCC MATLAB]:<https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab>
   
[char-rnn]:<https://github.com/karpathy/char-rnn>
[Andrej Karpathy's blog]:<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>
[Christopher Olah's blog]:<http://colah.github.io/>
[LDC]:<https://www.ldc.upenn.edu/data-management/using>
[matio]:<https://github.com/tbeu/matio>

[Efficient Hardware Mapping of LSTM Neural Networks for Speech Recognition]:<http://www.google.com>
[ESAT-MICAS]: <http://www.esat.kuleuven.be/micas/>



