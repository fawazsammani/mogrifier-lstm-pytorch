### Implementation of Mogrifier LSTM Cell in PyTorch
This follows the implementation of a Mogrifier LSTM proposed [here](https://arxiv.org/pdf/1909.01792.pdf)

The Mogrifier LSTM is an LSTM where two inputs `x` and `h_prev` modulate one another in an alternating fashion before the LSTM computation.

![Capture](https://user-images.githubusercontent.com/30661597/71353181-437f2080-25b3-11ea-97e6-fd52c796ad64.PNG)

You can easily define the Mogrifier LSTMCell just like defining nn.LSTMCell, with an additional parameter of `mogrify_steps`:
```python
mog_lstm = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
```

Here we provide an example of a model with two-layer Mogrifier LSTM. Additionally, the paper uses:
- Tying embedding weights and output weights
- Adam optimizer with betas = (0, 0.999), which essentially means ignoring the momentum term in Adam

The above points are implemented below as well.

```python
from mog_lstm import MogrifierLSTMCell
import torch
import torch.nn as nn
        
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)
        if tie_weights:
            self.fc.weight = self.embedding.weight
        
    def forward(self, seq, max_len = 10):
        
        embed = self.embedding(seq)
        batch_size = seq.shape[0]
        h1,c1 = [torch.zeros(batch_size,self.hidden_size), torch.zeros(batch_size,self.hidden_size)]
        h2,c2 = [torch.zeros(batch_size,self.hidden_size), torch.zeros(batch_size,self.hidden_size)]
        hidden_states = []
        outputs = []
        for step in range(max_len):
            x = self.drop(embed[:, step])
            h1,c1 = self.mogrifier_lstm_layer1(x, (h1, c1))
            h2,c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.fc(self.drop(h2))
            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))
            

        hidden_states = torch.cat(hidden_states, dim = 1)   # (batch_size, max_len, hidden_size)
        outputs = torch.cat(outputs, dim = 1)               # (batch_size, max_len, vocab_size)
        
        return outputs, hidden_states 
```

```python
input_size = 512
hidden_size = 512
mogrify_steps = 5
vocab_size = 30  
dropout = 0.5   # for simplicity: input dropout and output_dropout are 0.5. See paper for exact values
tie_weights = True
lr = 3e-3
betas = (0, 0.999)
weight_decay = 2.5e-4
clip_norm = 10
batch_size = 4
model = Model(input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=weight_decay)

# seq of shape (batch_size, max_words)
seq = torch.LongTensor([[ 8, 29, 18,  1, 17,  3, 26,  6, 26,  5],
                        [ 8, 28, 15, 12, 13,  2, 26, 16, 20,  0],
                        [15,  4, 27, 14, 29, 28, 14,  1,  0,  0],
                        [20, 22, 29, 22, 23, 29,  0,  0,  0,  0]])
                        
outputs, hidden_states = model(seq)
print(outputs.shape)
print(hidden_states.shape)
```

### Factorization of Q and R as products of low-rank matrices
Factorization of Q and R as products of low-rank matrices is not implemented. But you can implement it as follows:

```python
m = 512 
k = 85  # if set to 85: (512 * 85) + (85 * 512) << (512 * 512)
n = 512  

q_left = nn.Parameter(torch.FloatTensor(m, k).uniform_(-math.sqrt(1/n)/2, math.sqrt(1/n)/2))
q_right = nn.Parameter(torch.FloatTensor(k, n).uniform_(-math.sqrt(1/n)/2, math.sqrt(1/n)/2))
q = torch.matmul(q_left, q_right)
```
Then you can create a ModuleList to include all of them

