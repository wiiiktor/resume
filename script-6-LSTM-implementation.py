# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import re
import time
from time import ctime
import re
import torch
import numpy as np

# Configuration
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

# Set deterministic behaviour (only during development)
FIX_DETERMINISM = False
if FIX_DETERMINISM:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False

# Debugging setting
torch.autograd.set_detect_anomaly(False)


# Encoding logic
def load_and_encode(data_file):
    with open(data_file, 'r', errors="replace") as file:
        text = file.read()
        text = re.sub("([^\x00-\x7F])+", " ", text)

    chars = tuple(sorted(set(text)))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    encoded = np.array([char2int[ch] for ch in text])
    decoded = np.array([int2char[ch] for ch in encoded])

    return text, chars, encoded, decoded


text, chars, encoded, decoded = load_and_encode(data_file)

# Print details
print('chars:', chars)
print('chars length:', len(chars))
print('TEXT SAMPLE:\n', text[:30])
print('encoded:', encoded[:15])
print('decoded:', decoded[:15])


def one_hot_encode(arr, n_labels):
    """One-hot encodes the given array."""
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, n_seqs, n_steps, epoch=0):
    """Generates batches for training."""
    offset = epoch % n_steps
    arr = arr[offset:]

    # Compute batch size and number of batches
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    # Trim array to have full batches only
    arr = arr[:n_batches * batch_size]
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)

        y[:, :-1], y[:, -1] = x[:, 1:], arr[:, (n + n_steps) % arr.shape[1]]

        yield x, y


# Testing the batching function
batches = get_batches(encoded, 5, 10)
x, y = next(batches)

print('\nx\n', x)
print('\ny\n', y)


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
            hA, cA = self._lstm_cell(inputs[seq_idx].view(-1, len(self.chars)), hA, cA, self.gateA1, self.gateA2)

            # layer B
            hB, cB = self._lstm_cell(hA, hB, cB, self.gateB1, self.gateB2)

            outputs.append(hB)

        x = torch.stack(outputs)
        x = self.dropout(x)
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)
        x = self.fc(x)

        return x, ([hA, cA], [hB, cB])

    def _lstm_cell(self, input_tensor, h_prev, c_prev, gate1, gate2):
        gates = gate1(input_tensor)
        gates += gate2(h_prev)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_curr = (forgetgate * c_prev) + (ingate * cellgate)
        h_curr = outgate * torch.tanh(c_curr)

        return h_curr, c_curr

    def predict(self, char, h=None, top_k=None):
        """Predict the next character given a character."""

        if top_k is None:
            top_k = getattr(self, "top_k", 5)

        # Move the model to GPU if available
        self.cuda()

        # Initialize hidden state if not provided
        if h is None:
            h = self.init_hidden(1)

        h = tuple([[each[0].data, each[1].data] for each in h])

        # Convert char to input tensor
        input = np.array([[self.char2int[char]]])
        input = one_hot_encode(input, len(self.chars))
        input = torch.from_numpy(input).cuda()

        # Forward pass
        out, h = self.forward(input, h)
        p = F.softmax(out, dim=1).data.cpu()

        # Get top k characters
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()

        # Select the next character probabilistically
        char = np.random.choice(top_ch, p=p / p.sum())

        return self.int2char[char], h

    def init_weights(self):
        """Initialize weights for fully connected layer."""
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        """Initialize hidden and cell states."""
        weight = next(self.parameters()).data
        hidden = weight.new_zeros(self.n_layers, n_seqs, self.n_hidden)
        cell = weight.new_zeros(self.n_layers, n_seqs, self.n_hidden)
        return hidden, cell



def to_cuda(tensor):
    return tensor.cuda()


def one_hot_to_cuda(input, n_chars):
    input_encoded = one_hot_encode(input, n_chars)
    return to_cuda(torch.from_numpy(input_encoded))


def init_hidden_cuda(net, n_seqs):
    return tuple([to_cuda(each.data) for each in net.init_hidden(n_seqs)])


def calculate_accuracy(output, target, n_seqs, n_steps):
    _, top_class = torch.max(output, dim=1)
    correct_tensor = torch.eq(top_class, target.view(n_seqs * n_steps).type(torch.cuda.LongTensor))
    return torch.sum(correct_tensor).item()


def print_stats(epoch, epochs, counter, loss, val_losses, val_ok, val_ok_total):
    print('###################################################################################\n',
          f"Epoch: {epoch}/{epochs}...",
          f"Step: {counter}...",
          f"Loss: {loss:.4f}...",
          f"Val Loss: {np.mean(val_losses):.4f}",
          f"Val Accuracy: {val_ok / val_ok_total:.4f}",
          '\n###################################################################################')


def train(net, data, epochs, n_seqs, n_steps, lr, clip, val_frac, print_every, top_k):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    net.cuda()
    counter = 0
    n_chars = len(net.chars)

    for e in range(epochs):
        h = init_hidden_cuda(net, n_seqs)

        for input, target in get_batches(data, n_seqs, n_steps, e):
            counter += 1

            input, target = one_hot_to_cuda(input, n_chars), to_cuda(
                torch.from_numpy(target).view(n_seqs * n_steps).type(torch.cuda.LongTensor))

            h = init_hidden_cuda(net, n_seqs)

            net.zero_grad()
            output, _ = net.forward(input, h)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            if counter % print_every == 0:
                net.eval()

                val_h = init_hidden_cuda(net, n_seqs)
                val_losses = []
                val_ok, val_ok_total = 0, 0

                for input, target in get_batches(val_data, n_seqs, n_steps):
                    input, target = one_hot_to_cuda(input, n_chars), to_cuda(torch.from_numpy(target))
                    val_h = init_hidden_cuda(net, n_seqs)
                    output, _ = net.forward(input, val_h)
                    val_ok += calculate_accuracy(output, target, n_seqs, n_steps)
                    val_ok_total += n_steps
                    val_loss = criterion(output, target.view(n_seqs * n_steps).type(torch.cuda.LongTensor))
                    val_losses.append(val_loss.item())

                print_stats(e + 1, epochs, counter, loss.item(), val_losses, val_ok, val_ok_total)

                # Fixed batch testing
                h_sample = init_hidden_cuda(net, 1)
                chars_ = [ch for ch in fixed_prime]
                for ch in fixed_prime: char, h_sample = net.predict(ch, h_sample, top_k=top_k)
                chars_.append(char)

                for ii in range(fixed_sample_size):
                    char, h_sample = net.predict(chars_[-1], h_sample, top_k=top_k)
                    chars_.append(char)

                print(''.join(chars_), '\n')

                checkpoint = {
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'state_dict': net.state_dict(),
                    'tokens': net.chars
                }
                with open(model_name, 'wb') as f:
                    torch.save(checkpoint, f)

                net.train()


def initialize_network(load_file=None, tokens=None, hidden_layer=None, num_layers=None, drop_prob=None):
    if 'net' in locals():
        del net

    if load_file:
        with open(load_file, 'rb') as f:
            checkpoint = torch.load(f)
        net = Main(tokens=checkpoint['tokens'], n_hidden=checkpoint['n_hidden'])
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net = Main(tokens=tokens, n_hidden=hidden_layer, n_layers=num_layers, drop_prob=drop_prob)

    return net

def sample(net, size=2000, prime=' ', top_k=5):
    net.cuda()
    net.eval()

    chars = [ch for ch in prime]
    h = net.init_hidden(1)

    for ch in prime:
        char, h = net.predict(ch, h, top_k=top_k)
        chars.append(char)

    for _ in range(size):
        char, h = net.predict(chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

# Main Execution
net = initialize_network(load_file=load_file if loading else None, tokens=chars, hidden_layer=hidden_layer, num_layers=num_layers, drop_prob=drop_prob)
print('\n', net, '\n')

print(ctime(time.time()))
train(net, encoded, epochs=epochs, n_seqs=num_seqs, n_steps=num_steps, lr=0.001,
      print_every=print_every, clip=5, val_frac=0.1, top_k=top_k)
print(ctime(time.time()))

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
