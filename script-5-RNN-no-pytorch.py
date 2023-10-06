"""
Minimal character-level Vanilla RNN model. 
Written by Andrej Karpathy (@karpathy).
The second layer added by Wiktor Migaszewski.
BSD License
"""
import numpy as np

# ==== DATA I/O ====
file_path = 'data/40.txt'

with open(file_path, 'r') as file:
    data = file.read()

chars = sorted(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# ==== HYPERPARAMETERS ====
print_every = 500
hidden_size = 100
seq_length = 25
learning_rate = 0.005

# ==== MODEL PARAMETERS ====
def initialize_weights(rows, cols):
    return np.random.randn(rows, cols) * 0.01

def initialize_bias(size):
    return np.zeros((size, 1))

# Layer 1
Wxh = initialize_weights(hidden_size, vocab_size)  # input to hidden
Whh = initialize_weights(hidden_size, hidden_size)  # hidden to hidden
bh = initialize_bias(hidden_size)

# Layer 2
Whh2 = initialize_weights(hidden_size, hidden_size)  # hidden to hidden2
Wh2h2 = initialize_weights(hidden_size, hidden_size)  # hidden2 to hidden2
bh2 = initialize_bias(hidden_size)

# Output Layer
Wh2y = initialize_weights(vocab_size, hidden_size)  # hidden2 to output
by = initialize_bias(vocab_size)



def lossFun(inputs, targets, hprev, h2prev):
    """
    Computes the loss, gradients on model parameters, and last hidden states.

    Parameters:
    - inputs, targets: Lists of integers.
    - hprev, h2prev: Initial hidden states (Hx1 and H2x1 arrays respectively).

    Returns:
    - loss, gradients on model parameters, and last hidden states.
    """
    xs, hs, h2s, ys, ps = {}, {}, {}, {}, {}
    hs[-1], h2s[-1] = np.copy(hprev), np.copy(h2prev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        xs[t] = create_one_hot(inputs[t])
        hs[t] = compute_hidden_state(Wxh, xs[t], Whh, hs[t-1], bh)
        h2s[t] = compute_hidden_state(Whh2, hs[t], Wh2h2, h2s[t-1], bh2)
        ys[t], ps[t] = compute_output(h2s[t])

        loss += -np.log(ps[t][targets[t], 0])  # Cross-entropy loss

    # Initialize gradients
    dWxh, dWhh, dWhh2, dWh2h2, dWh2y = [np.zeros_like(param) for param in [Wxh, Whh, Whh2, Wh2h2, Wh2y]]
    dbh, dbh2, dby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(by)
    dhnext, dh2next = np.zeros_like(hs[0]), np.zeros_like(h2s[0])

    # Backward pass
    for t in reversed(range(len(inputs))):
        dWh2y, dby, dh2next = backprop_output(ps[t], targets[t], h2s[t], dh2next)
        dWhh2, dhnext = backprop_hidden(dh2next, h2s[t-1], h2s[t], Wh2h2, Whh2, hs[t])
        dWxh, dWhh, dhnext = backprop_input(dhnext, hs[t-1], xs[t], hs[t], Whh)

    # Clip gradients to mitigate exploding gradients
    for dparam in [dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby, hs[len(inputs) - 1], h2s[len(inputs) - 1]


def create_one_hot(index):
    vec = np.zeros((vocab_size, 1))
    vec[index] = 1
    return vec


def compute_hidden_state(W1, x, W2, h_prev, b):
    return np.tanh(np.dot(W1, x) + np.dot(W2, h_prev) + b)


def compute_output(h2):
    y = np.dot(Wh2y, h2) + by
    p = np.exp(y) / np.sum(np.exp(y))
    return y, p


def backprop_output(p, target, h2, dh2next):
    dy = np.copy(p)
    dy[target] -= 1
    dWh2y = np.dot(dy, h2.T)
    dby = dy
    dh2 = np.dot(Wh2y.T, dy) + dh2next
    return dWh2y, dby, dh2


def backprop_hidden(dh2next, h2prev, h2, Wh2h2, Whh2, h1):
    dh2raw = (1 - h2 * h2) * dh2next
    dbh2 = dh2raw
    dWh2h2 = np.dot(dh2raw, h2prev.T)
    dh2next = np.dot(Wh2h2.T, dh2raw)
    dWhh2 = np.dot(dh2raw, h1.T)
    return dWhh2, dh2next


def backprop_input(dhnext, hprev, x, h, Whh):
    dhraw = (1 - h * h) * dhnext
    dbh = dhraw
    dWhh = np.dot(dhraw, hprev.T)
    dhnext = np.dot(Whh.T, dhraw)
    dWxh = np.dot(dhraw, x.T)
    return dWxh, dWhh, dhnext



def sample(h, h2, seed_ix, n):
    """
    Sample a sequence of integers from the model.
    - h, h2: Initial memory states.
    - seed_ix: Seed letter for the first time step.
    - n: Number of integers to sample.

    Returns:
    - A list of sampled integers.
    """
    x = create_one_hot(seed_ix)
    ixes = []

    for _ in range(n):
        h, h2 = forward_step(h, h2, x)
        ix = sample_next_character(h2)
        x = create_one_hot(ix)
        ixes.append(ix)

    return ixes

def forward_step(h, h2, x):
    """Execute a forward step of the model."""
    h = compute_hidden_state(Wxh, x, Whh, h, bh)
    h2 = compute_hidden_state(Whh2, h, Wh2h2, h2, bh2)
    return h, h2

def sample_next_character(h2):
    """Sample the next character based on the output distribution."""
    y = compute_output_without_softmax(h2)
    p = softmax(y)
    return np.random.choice(range(vocab_size), p=p.ravel())

def create_one_hot(index):
    """Return a one-hot encoded vector."""
    vec = np.zeros((vocab_size, 1))
    vec[index] = 1
    return vec

def compute_output_without_softmax(h2):
    """Compute the output of the model without applying softmax."""
    return np.dot(Wh2y, h2) + by

def softmax(y):
    """Compute the softmax of a vector."""
    return np.exp(y) / np.sum(np.exp(y))


def initialize_memory_variables():
    """Initialize memory variables for Adagrad."""
    mWxh, mWhh, mWhh2, mWh2h2, mWh2y = [np.zeros_like(param) for param in [Wxh, Whh, Whh2, Wh2h2, Wh2y]]
    mbh, mbh2, mby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(by)
    return mWxh, mWhh, mWhh2, mWh2h2, mWh2y, mbh, mbh2, mby


def adagrad_update(params, dparams, mems, learning_rate):
    """Perform parameter update with Adagrad."""
    for param, dparam, mem in zip(params, dparams, mems):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
    return params, mems


def adjust_learning_rate(n, learning_rate):
    """Lower the learning rate at specific iteration counts."""
    adjustment_points = {20_000, 100_000, 400_000, 1_000_000}
    if n in adjustment_points:
        learning_rate /= 2
        print(learning_rate)
    return learning_rate


# Main Training Loop
n, p = 0, 0
mWxh, mWhh, mWhh2, mWh2h2, mWh2y, mbh, mbh2, mby = initialize_memory_variables()
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

while True:  # loop indefinitely
    # Reset or progress data pointer
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        h2prev = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # Sample periodically
    if n % print_every == 0:
        sample_ix = sample(hprev, h2prev, inputs[0], 150)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f'----\n{txt}\n----')
        print(f'iter {n}, loss: {smooth_loss}')

    # Compute loss and gradients
    loss, dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby, hprev, h2prev = lossFun(inputs, targets, hprev, h2prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # Adjust learning rate
    learning_rate = adjust_learning_rate(n, learning_rate)

    # Update model parameters with Adagrad
    params = [Wxh, Whh, Whh2, Wh2h2, Wh2y, bh, bh2, by]
    grads = [dWxh, dWhh, dWhh2, dWh2h2, dWh2y, dbh, dbh2, dby]
    mems = [mWxh, mWhh, mWhh2, mWh2h2, mWh2y, mbh, mbh2, mby]

    params, mems = adagrad_update(params, grads, mems, learning_rate)

    p += seq_length
    n += 1
