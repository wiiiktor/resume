# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import re
import time
from time import ctime

# fixing random seed, set to True only during development, otherwise you will always get the same results
if False:
    torch.manual_seed(0)
    # random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # dataloader also may require fixing
torch.autograd.set_detect_anomaly(False)  # set to True only for debugging

# z rzeczy nietypowych, skrypt zawiera fixed batch testing, load/save, accuracy
# podac przyklad sample po 100 epokach nauki z tekstu Szekspira 400 lines / 11300 chars
# kluczowy fragment kodu layer1/layer2 mozna wstawic do pliku README

data_file = '400.txt'
model_name = 'data/script-3---saved-model.net'
load_file = 'data/script-3---saved-model.net'
loading = True
epochs = 100
num_seqs = 1
num_steps = 32
num_layers = 2
hidden_layer = 128
top_k = 5
drop_prob = 0.5
print_every = 100
prime1 = 'before'
sample_size = 2000
fixed_prime = 'before'
fixed_sample_size = 50
torch.autograd.set_detect_anomaly(False)

# region encoding
file = open(data_file, 'r', errors="replace")
letters = []
for line in file:
    line = re.sub("([^\x00-\x7F])+", " ", line)
    for character in line:
        if character not in letters:
            letters.append(character)
print('letters', letters)

with open(data_file, 'r') as f: text = f.read()
chars = tuple(sorted(set(text)))
print('chars', chars)
print('chars length', len(chars))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])
decoded = np.array([int2char[ch] for ch in encoded])
print('TEXT SAMPLE:\n', text[:30])
print('encoded:', encoded[:15])
print('decoded:', decoded[:15])


def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, n_seqs, n_steps, epoch=0):
    x = epoch % n_steps
    arr = arr[x:]
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


batches = get_batches(encoded, 5, 10)
x, y = next(batches)
print('\nx\n', x[:])
print('\ny\n', y[:])
# endregion

class Main(nn.Module):
    def __init__(self, tokens, n_hidden, n_layers, drop_prob, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.gateA1 = nn.Linear(len(self.chars), n_hidden * 4)
        self.gateA2 = nn.Linear(n_hidden, n_hidden * 4)

        self.gateB1 = nn.Linear(n_hidden, n_hidden * 4)
        self.gateB2 = nn.Linear(n_hidden, n_hidden * 4)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

        self.init_weights()

    def forward(self, x, hc):
        hA, cA = hc[0]
        hB, cB = hc[1]
        inputs = x.unbind(0)[0]
        outputs = []

        for seq_idx in range(len(inputs)):

            # layer A
            gates = self.gateA1(inputs[seq_idx].view(-1, len(self.chars)))
            gates += self.gateA2(hA.view(-1, hidden_layer))

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cA = (forgetgate * cA) + (ingate * cellgate)
            hA = outgate * torch.tanh(cA)

            # layer B
            gates = self.gateB1(hA)
            gates += self.gateB2(hB)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cB = (forgetgate * cB) + (ingate * cellgate)
            hB = outgate * torch.tanh(cB)

            outputs += [hB]

        x = torch.stack(outputs)
        x = self.dropout(x)
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)
        x = self.fc(x)

        return x, ([hA, cA], [hB, cB])

    def predict(self, char, h=None, top_k=top_k):
        self.cuda()

        if h is None: h = self.init_hidden(1)
        h = tuple([[each[0].data, each[1].data] for each in h])

        input = np.array([[self.char2int[char]]])
        input = one_hot_encode(input, len(self.chars))
        input = torch.from_numpy(input).cuda()

        out, h = self.forward(input, h)
        p = F.softmax(out, dim=1).data

        p = p.cpu()
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        return self.int2char[char], h

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())


def train(net, data, epochs, n_seqs, n_steps, lr, clip, val_frac, print_every, top_k):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    net.cuda()
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(n_seqs)

        for input, target in get_batches(data, n_seqs, n_steps, e):
            counter += 1

            input = one_hot_encode(input, n_chars)
            input = torch.from_numpy(input).cuda()
            target = torch.from_numpy(target).cuda().view(n_seqs * n_steps).type(torch.cuda.LongTensor)

            # Creating new h, to avoid doing backprop through the entire training history
            h = tuple([[each[0].data, each[1].data] for each in h])

            net.zero_grad()
            output, _ = net.forward(input, h)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            if counter % print_every == 0:
                net.eval()
                val_h = net.init_hidden(n_seqs)
                val_losses, val_ok, val_ok_total = [], 0, 0
                for input, target in get_batches(val_data, n_seqs, n_steps):
                    # One-hot encode our data and make them Torch tensors
                    input = one_hot_encode(input, n_chars)
                    input, target = torch.from_numpy(input), torch.from_numpy(target)

                    val_h = tuple([[each[0].data, each[1].data] for each in val_h])

                    input, target = input.cuda(), target.cuda()

                    output, _ = net.forward(input, val_h)

                    vall, indd = torch.max(output, dim=1)
                    ok = torch.eq(indd, target.view(n_seqs * n_steps).type(torch.cuda.LongTensor))
                    val_ok += torch.sum(ok)
                    val_ok_total += num_steps
                    val_loss = criterion(output, target.view(n_seqs * n_steps).type(torch.cuda.LongTensor))
                    val_losses.append(val_loss.item())

                print('###################################################################################\n',
                      "Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)),
                      "Val Accuracy:  {:.4f}".format(val_ok / val_ok_total),
                      '\n###################################################################################')
                losses, val_ok, val_ok_total = [], 0, 0

                # Fixed batch testing
                h_sample = net.init_hidden(1)
                chars_ = [ch for ch in fixed_prime]
                for ch in fixed_prime: char, h_sample = net.predict(ch, h_sample, top_k=top_k)
                chars_.append(char)

                # Now pass in the previous character and get a new one
                for ii in range(fixed_sample_size):
                    char, h_sample = net.predict(chars_[-1], h_sample, top_k=top_k)
                    chars_.append(char)

                print(''.join(chars_), '\n')
                net.train()

                checkpoint = {'n_hidden': net.n_hidden,
                              'n_layers': net.n_layers,
                              'state_dict': net.state_dict(),
                              'tokens': net.chars}

                with open(model_name, 'wb') as f:
                    torch.save(checkpoint, f)


if 'net' in locals(): del net

# Initialize and print the network
net = Main(chars, n_hidden=hidden_layer, n_layers=num_layers, drop_prob=drop_prob)
print('\n', net, '\n')

if loading:
    with open(load_file, 'rb') as f: checkpoint = torch.load(f)
    net = Main(tokens=checkpoint['tokens'], n_hidden=checkpoint['n_hidden'])
    net.load_state_dict(checkpoint['state_dict'])

print(ctime(time.time()))
train(net, encoded, epochs=epochs, n_seqs=num_seqs, n_steps=num_steps, lr=0.001,
      print_every=print_every, clip=5, val_frac=0.1, top_k=top_k)
print(ctime(time.time()))

def sample(net, size=2000, prime=' ', top_k=5):
    net.cuda()
    net.eval()

    # First off, run through the prime characters
    chars = [ch for ch in prime]

    h = net.init_hidden(1)

    for ch in prime: char, h = net.predict(ch, h, top_k=top_k)
    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = net.predict(chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print('################\n', sample(net, sample_size, prime=prime1, top_k=top_k))


''' RESULT EXAMPLE: 
###################################################################################
 Epoch: 100/100... Step: 31500... Loss: 0.4776... Val Loss: 3.4674 Val Accuracy:  0.4107 
###################################################################################
before me recominr good words to thee will flatter
Beneat 

###################################################################################
 Epoch: 100/100... Step: 31600... Loss: 0.4090... Val Loss: 3.6070 Val Accuracy:  0.4071 
###################################################################################
before receive the fartiend comporan ffecendmnd the me, t 

###################################################################################
 Epoch: 100/100... Step: 31700... Loss: 0.4479... Val Loss: 3.5866 Val Accuracy:  0.4089 
###################################################################################
before repetl heard ofinly, to the seorat of the whole bo 

Thu Jun 16 18:48:29 2022
################
 before mekent,
The seak thit
Corn for the rich men only: with these shreds
They vented their cares, digest they art upot in receive
But it pactice, who discrements are
That unconnot
See what I do deliver out to each,
Yet I can make my aunit up, poode.

First Citizen:
It was an answer: body's made for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
supporter in the horns o' the moon,
Shouting their emigh? belly,
And you the moty sing cowardly. But, I beseech you,
Where huse they will; levedese they whas an on the howserving. I why twas don these low speaks! What shouds with meir but the how recome rather that I shall tell you
I this make lice candeded complain,
What could the commonalty.

Second Citizen:
What he cannot ees, the garte do reat our senature
With the rest, what services he has done fatoess,
Whose but the Capitol; who's like to rain their liking
Bearing
Lith these shered they lack discretion,
Yet are they will; and
there wollh as antigut apms, the hobl fo the common, you saye their own rates; whereof, they say,
The city is well stored.

MARCIUS:
Hang 'em! They say!
They'll sit to the people.

All:
We lood, neve sence it well, seral shouts aroutts,
That, luaved us than says the other troog
Or be arch you corlinc out
is,
Bheir own choice: one's Junius Brutus,
Sicinius Velutus, and I know not--'Sdeath!
The rabble should have first unroof'd the city,
Ere so prevail'd with me what services he has done for his country hase forthights and gats mese lincesence are thorest le body. The belly answer?;
Where go you,
Where ho services of loke a there while on'th  own from them to you
And no way from yourselvis. What sorvest ast ffricts of youe opitizen:
Sorthy proud; which he
Thank now't  arts their ruth,
And let me use my sword, I's make sigh'd poorse fribfdded their vuugarse that we and we'll say we proceeds on owes arch agest their rotd,
not eal in rich ang their carss,
When you do live upon; and fit in up, these a there 
'''
