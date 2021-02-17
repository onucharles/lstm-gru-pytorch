# Implementations of LSTM and GRU in Pytorch

Implementation of recurrent neural networks (RNNs) from "scratch" in PyTorch. The only PyTorch module used is 
nn.Linear. 

I had to write this for [a research project](https://github.com/onucharles/tensorized-rnn). I needed to make 
internal changes to RNNs for my experiments but observed that PyTorch's RNNs were imported as C libraries. Hopefully 
this will save you a few hours or days in your own work :-)  

RNNS implemented are:
* Long short-term memory, LSTM
* Gated recurrent Unit, GRU


## Dependency
Torch