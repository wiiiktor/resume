This repository contains a few AI srcipts that I wrote:
1. CNN working with batch_size=1
2. RNN 2-layer, implemented with raw Python & NumPy, with derivatives calculation
3. LSTM 2-layer, with no use of nn.LSTM PyTorch object

It also contains my full CV, but here in short words: I am a graduate of the Warsaw School of Economics, with 10+ years of work experience of a business analyst for a few corporations (Masterfoods, METRO Group, Whirlpool). I had also an episode of [work as PHP programmer](https://github.com/wiiiktor/resume/blob/main/Wiktor_Migaszewski_referencje_serwisu_konsumenckiego_www_Bazaria_pl.pdf) years ago; apart from that I only had professional experience in scarce scirpting of Microsoft tools or specialised applications. Still, I wrote code in PHP, C#, JAVA and Python, for my hobby projects. Over the last years, I worked in a family company Efneo.com, operating on a bicycle market. Currently, I develop my interest in AI / Machine Learning and would like to enter this area to gain prefessional experience and create my own AI company in the future.

## Ad 1. CNN working with batch_size=1
Network reaches TOP-1 Accuracy of 77% and TOP-5 Accuracy of 93%, without using Batch Normalisation. I had a reason to avoid using BN, as I needed batch_size=1. My application aimed at running subnetworks for various tasks, for example if initial detection was "an animal", a subnetwork "recognizeAnimals" would be used, but in case of initial detection of "a car", a subnetwork "recognizeCarModels" would be used. This concept requires using conditions depending on the initial results that triggered different paths through the network. Well, with a typical batch_size=32 I would not be albe to do it, as batch would have to be splat into different subnetworks... This is why I needed a method working with batch_size=1.

I used a solution described in here: https://arxiv.org/pdf/1903.10520.pdf / https://youtu.be/m3TN9FFmqsI To be more specific, an option of GN+WS (Group Normalization + Weight Standarization). Key code fragment: 
```{python}
class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, 
                                       kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
```
CNN now includes layers of: 
```{python}
self.Conv2dWS()
self.GroupNorm()
```
instead of standard ones: 
```{python}
self.Conv2d()
self.BatchNorm()
```

## Ad 2. RNN 2-layer, implemented with raw Python & NumPy
RNN working even for longer fragments of text, with no use of PyTorch nn.RNN object, but only Python + NumPy. Sript is a modification of Andrey Karpathy code, by adding the 2. layer (thus making is capable of learning from much reacher text databases) and gradual decreasing learning rate. Script calculates derivatives and even implements Adagrad optmiser. Below you can find a sample of text generated after learning from 40 lines of Sheakspeare:

```
Second Citizen:
One word, good citizens.
First Citizen:
Let us kill him, and we'll have corn  
 omish?
SlfancoC
ouu kencekiss soke, shel aememis 
```
This is the only import line in this script :smiley:
```{python}
import numpy as np
```

## Ad 3. LSTM 2-layer with no use of nn.LSTM PyTorch object
Script implements all the LSTM operations defined below:


<img src=https://i.stack.imgur.com/L6W94.png>


Key code fragment: 
```{python}
ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
ingate = torch.sigmoid(ingate)
forgetgate = torch.sigmoid(forgetgate)
cellgate = torch.tanh(cellgate)
outgate = torch.sigmoid(outgate)
cA = (forgetgate * cA) + (ingate * cellgate)
hA = outgate * torch.tanh(cA)
```
