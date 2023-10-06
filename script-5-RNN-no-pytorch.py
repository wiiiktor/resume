"""
Minimal character-level Vanilla RNN model. 
The second layer added by Wiktor Migaszewski, 
based on one-layer script by Andrej Karpathy (@karpathy).
BSD License
"""
import numpy as np

# data I/O
data = open('data/40.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has ' + str(data_size) + ' characters, ' + str(vocab_size) + ' unique.')
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
print_every = 500
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 0.005

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
bh = np.zeros((hidden_size, 1))  # hidden bias

Whh2 = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden2
Wh2h2 = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden2 to hidden2
bh2 = np.zeros((hidden_size, 1))  # hidden2 bias

Wh2y = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden2 to output
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev, h2prev):
    """
    inputs,targets are both list of integers
    hprev is Hx1 array of initial hidden state
    h2prev is H2x1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, h2s, ys, ps = {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    h2s[-1] = np.copy(h2prev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)  # hidden state
        h2s[t] = np.tanh(np.dot(Whh2, hs[t]) + np.dot(Wh2h2, h2s[t-1]) + bh2)  # hidden2 state
        ys[t] = np.dot(Wh2y, h2s[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh = np.zeros_like(Wxh), np.zeros_like(Whh)
    dWhh2, dWh2h2 = np.zeros_like(Whh2), np.zeros_like(Wh2h2)
    dWh2y = np.zeros_like(Wh2y)
    dbh, dbh2, dby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    dh2next = np.zeros_like(h2s[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad
        dWh2y += np.dot(dy, h2s[t].T)
        dby += dy

        dh2 = np.dot(Wh2y.T, dy) + dh2next  # backprop into h2
        dh2raw = (1 - h2s[t] * h2s[t]) * dh2  # backprop through tanh nonlinearity
        dbh2 += dh2raw
        dWh2h2 += np.dot(dh2raw, h2s[t-1].T)
        dh2next = np.dot(Wh2h2.T, dh2raw)

        dWhh2 += np.dot(dh2raw, hs[t].T)

        dh = np.dot(Whh2.T, dh2) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

        dWxh += np.dot(dhraw, xs[t].T)

    for dparam in [dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby, hs[len(inputs) - 1], h2s[len(inputs) - 1]


def sample(h, h2, seed_ix, n):
    """
    sample a sequence of integers from the model
    h, h2 are memory states, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h =  np.tanh(np.dot(Wxh,  x) + np.dot(Whh,   h ) + bh )
        h2 = np.tanh(np.dot(Whh2, h) + np.dot(Wh2h2, h2) + bh2)
        y = np.dot(Wh2y, h2) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhh2, mWh2h2, mWh2y = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Whh2), np.zeros_like(Wh2h2), np.zeros_like(Wh2y)
mbh, mbh2, mby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:  # we loop indefinitely
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        h2prev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p + seq_length+1]]

    # sample from the model now and then
    if n % print_every == 0:
        sample_ix = sample(hprev, h2prev, inputs[0], 150)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n ' + txt + ' \n----')

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby, hprev, h2prev = lossFun(inputs, targets, hprev, h2prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % print_every == 0: print('iter ' + str(n) + ', loss: ' + str(smooth_loss))   # print progress

    # lowering the learning rate over time
    if n == 20_000: learning_rate /= 2; print(learning_rate)
    if n == 100_000: learning_rate /= 2; print(learning_rate)
    if n == 400_000: learning_rate /= 2; print(learning_rate)
    if n == 1_000_000: learning_rate /= 2; print(learning_rate)
        
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([ Wxh,  Whh,  Whh2,  Wh2h2,  Wh2y,  bh,  bh2,  by],
                                  [dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby],
                                  [mWxh, mWhh, mWhh2, mWh2h2, mWh2y, mbh, mbh2, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter
