Repozytorium zawiera następujące skrypty AI mojego autorstwa:
<ol><li>CNN działający z batch_size=1</li>
<li>RNN 2-warstwowy zaimplementowany z użyciem "czystego" Pythona & NumPy, z liczeniem pochodnych</li>
<li>LSTM 2-warstwowy zaimplementowany bez użycia obiektu nn.LSTM PyTorcha</li></ol>

<h2>Ad 1. CNN działający z batch_size=1</h2>
Sieć uzysktuje wynik Accuracy TOP-1 77% oraz Accuracy TOP-5 ..., czyli SOTA, bez wykorzystania Batch Normalization. Musiałem uniknąć stosowania BN, żeby móc użyć batch_size=1, ponieważ moim planem było stworzenie sieci zawierającej wewnętrzne warunki, uruchamiające kolejne podsieci. Innymi słowy, chciałem stworzyć kod, uruchamiający podsieć dedykowaną dla zwierząt, jeśli wykryte zostanie zwierze (if result==Animal then run subnetwork recognizeAnimals). Oczywiście, jeśli stosowałbym batch_size=32, to sieć musiałaby obsłużyć 32 różne obiekty... a to byłyby zwierzęta, samochody, budynki, etc, w jednym batch'u. Batch musiałby się "rozjechać", a PyTorch oczywiście nie posiada takicej funkcjonalności. Dlatego musiałbem znaleźć rozwiązanie umożliwjające trening przy batch_size=1.  

Rozwiązaniem okazało się podejście opisane tutaj: https://arxiv.org/pdf/1903.10520.pdf / https://youtu.be/m3TN9FFmqsI a dokładnie wariant GN+WS (Group Normalization + Weight Standarization). Kluczowy fragment kodu znajduje się poniżej: 
```{python}
Conv2dWS(nn.Module):....
```
Sieć CNN zawiera warstwy: 
```{python}
self.Conv2dWS()
self.GroupNorm()
```
zamiast standardowej pary: 
```{python}
self.Conv2d()
self.BatchNorm()
```

<h2>Ad 2. RNN 2-warstwowy, z użyciem "czystego" Pythona & NumPy</h2>
Sieć RNN działająca stabilnie nawet dla dłuższych tekstów, bez wykorzystania obiektu nn.RNN czy nn.LSTM PyTorcha, jedynie czystego Pythona + NumPy. Skrypt jest modyfikacją kodu Andrej'a Karpathy'ego, poprzez dodanie 2. warstwy, co znacznie zwiększa jego możliwości, oraz stopniowe zmniejszanie learning_rate. Skrypt samodzielnie liczy pochodne oraz stany pamięci dla optymalizatora Adagrad. Poniżej przykładowy tekst wygenerowany po nauce na 400 liniach dramatu Szekspira: 
```
 irst Citizen:
Before we proceed any further, hear  
```
Jedyny import tego skryptu: 
```{python}
import numpy as np
```

<h2>Ad 3. LSTM 2-warstwowy, bez użycia obiektu nn.LSTM PyTorcha</h2>
Skrypt stanowi implementację LSTM zawierającą wszystkie operacje określone poniżej: 
<br><img src='https://i.stack.imgur.com/L6W94.png'></img><br>
Kluczowy kod: 
```{python}
gates = ...
```
