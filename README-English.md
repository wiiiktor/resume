This repository contains a few AI srcipts that I wrote:
<ol><li>CNN working with batch_size=1</li>
<li>RNN 2-layer, implemented with raw Python & NumPy, with calculating derivatives</li>
<li>LSTM 2-layer with no use of nn.LSTM PyTorch object</li></ol>

<h2>Ad 1. CNN working with batch_size=1</h2>
Sieć uzysktuje wynik Accuracy TOP-1 77% oraz Accuracy TOP-5 ..., 
Network reaches TOP-1 Accuracy of 77% and TOP-5 Accuracy of ..., without using Batch Normalisation. I had a reason to avoid using BN, as I needed batch_size=1. My application aimed at running subnetworks for various tasks, for example if initial detection was "an animal", a subnetwork "recognizeAnimals" would be used, but in case of initial detection of "a car", a subnetwork "recognizeCarModels" would be used. This concept requires using conditions depending on the initial results that triggered different paths through the network. Well, with a typical batch_size=32 I would not be albe to do it, as batch would have to be splat into different subnetworks... This is why I needed a method working with batch_size=1.

I uses a solution described here: https://arxiv.org/pdf/1903.10520.pdf / https://youtu.be/m3TN9FFmqsI To be more specific, an option of GN+WS (Group Normalization + Weight Standarization). Key code fragment: 
```{python}
Conv2dWS(nn.Module):....
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

<h2>Ad 2. RNN 2-layer, implemented with raw Python & NumPy</h2>
Sieć RNN działająca stabilnie nawet dla dłuższych tekstów, bez wykorzystania obiektu nn.RNN czy nn.LSTM PyTorcha, jedynie czystego Pythona + NumPy. Skrypt jest modyfikacją kodu Andreja Karpathego, poprzez dodanie 2. warstwy, co znacznie zwiększa jego możliwości, oraz stopniowe zmniejszanie learning_rate. Skrypt samodzielnie liczy pochodne oraz stany pamięci dla optymalizatora Adagrad. Poniżej przykładowy tekst wygenerowany po nauce na 400 liniach dramatu Szekspira: 

```{python}
 irst Citizen:
Before we proceed any further, hear  
```
This is the only import line in this script :smiley:
```{python}
import numpy as np
```

<h2>Ad 3. LSTM 2-layer with no use of nn.LSTM PyTorch object</h2>
Script implements all the LSTM operations defined below:


<img src=https://i.stack.imgur.com/L6W94.png>


Key code fragment: 
```{python}
gates = ...
```
