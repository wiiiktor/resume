Repozytorium zawiera następujące skrypty AI mojego autorstwa:
CNN działający z batch_size=1
RNN 2-warstwowy zaimplementowany z użyciem "czystego" Pythona & NumPy
LSTM 2-warstwowy zaimplementowany bez użycia obiektu nn.LSTM PyTorcha 

Ad 1. Sieć uzysktuje wynik Accuracy TOP-1 77% oraz Accuracy TOP-5 ..., czyli SOTA, bez wykorzystania Batch Normalization. Musiałem uniknąć stosowania BN, żeby móc użyć batch_size=1, ponieważ moim planem było stworzenie sieci zawierającej wewnętrzne warunki, uruchamiające podsieci. Innymi słowy, chciałem stworzyć kod, uruchamiający podsieć dedykowaną dla zwierząt, jeśli wykryte zostanie zwierze (if animal then run subnetwork recognizeAnimals). Oczywiście, jeśli stosowałbym batch_size=32, to sieć musiałaby obsłużyć 32 różne obiekty... czasem byłyby to zwierzęta, a czasem samochody. Batch musiałby się "rozjechać", 

Chciałem uniknąć stosowania BN, żeby móc stosować batch_size=1, ponieważ chciałe. Rozwiązaniem okazało się podejście opisane tutaj: https://arxiv.org/pdf/1903.10520.pdf / https://youtu.be/m3TN9FFmqsI a dokładnie wariant GN+WS (Group Normalization + Weight Standarization). Kluczowy fragment kodu znajduje się poniżej: 

Conv2dWS(nn.Module):....
Sieć CNN zawiera warstwy: 
self.Conv2dWS()
self.GroupNorm()
zamiast standardowego zestawu: 
self.Conv2d()
self.BatchNorm()



, jedynie Python+NumPy. Kod liczy pochodne, a nawet zawiera implementację optymalizatora Adagrad. Podstawą był skrypt Karpathy'ego, 1-warstwowy, do którego dodałem drugą warstwę.
