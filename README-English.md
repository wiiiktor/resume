This repository contains a few AI srcipts that I wrote:
<ol><li>CNN working with batch_size=1</li>
<li>RNN 2-layer, implemented with raw Python & NumPy, with calculating derivatives</li>
<li>LSTM 2-layer with no use of nn.LSTM PyTorch object</li></ol>

<h2>Ad 1. CNN working with batch_size=1</h2>
Sieć uzysktuje wynik Accuracy TOP-1 77% oraz Accuracy TOP-5 ..., czyli SOTA, bez wykorzystania Batch Normalization. Musiałem uniknąć stosowania BN, żeby móc użyć batch_size=1, ponieważ moim planem było stworzenie sieci zawierającej wewnętrzne warunki, uruchamiające kolejne podsieci. Innymi słowy, chciałem stworzyć kod, uruchamiający podsieć dedykowaną dla zwierząt, jeśli wykryte zostanie zwierze (if result==Animal then run subnetwork recognizeAnimals). Oczywiście, jeśli stosowałbym batch_size=32, to sieć musiałaby obsłużyć 32 różne obiekty... a to byłyby zwierzęta, samochody, budynki, etc, w jednym batch'u. Batch musiałby się "rozjechać", a PyTorch oczywiście nie posiada takicej funkcjonalności. Dlatego musiałbem znaleźć rozwiązanie umożliwjające trening przy batch_size=1.

I use a solution described here: https://arxiv.org/pdf/1903.10520.pdf / https://youtu.be/m3TN9FFmqsI To be more specific, an option of GN+WS (Group Normalization + Weight Standarization). Key code fragment: 
```{python}
Conv2dWS(nn.Module):....
```
CNN includes layers of: 
```{python}
self.Conv2dWS()
self.GroupNorm()
```
instead of standard: 
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
This is the only import line in this script: 
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
