### Implementation of Mogrifier LSTM Cell in PyTorch
This follows the implementation of a Mogrifier LSTM proposed [here](https://arxiv.org/pdf/1909.01792.pdf)

The Mogrifier LSTM is an LSTM where two inputs x and h_prev modulate one another in an alternating fashion before the LSTM computation.

![Capture](https://user-images.githubusercontent.com/30661597/71353181-437f2080-25b3-11ea-97e6-fd52c796ad64.PNG)

```python
import torch
import torch.nn as nn
import math

class MogrifierLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.q = nn.Linear(hidden_size, input_size)
        self.r = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.init_parameters()
        self.mogrify_steps = mogrify_steps
    
    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)
            
    def mogrify(self, x, h):
        for i in range(1,self.mogrify_steps+1):
            if i % 2 == 0:
                h = (2*torch.sigmoid(self.r(x))) * h
            else:
                x = (2*torch.sigmoid(self.q(h))) * x
        return x, h

    def forward(self, x, states):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        """
        ht, ct = states
        x, ht = self.mogrify(x,ht)
        gates = self.x2h(x) + self.h2h(ht)    # (batch_size, 4 * hidden_size)
        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        h_new = out_gate * self.tanh(c_new)

        return h_new, c_new

```

The example below shows how you can use a mogrifier LSTM:

```
xt = torch.randn(5,512)    # input: (batch_size, input_size)
ht = torch.randn(5,512)    # hidden_state: (batch_size, hidden_size)
ct = torch.randn(5,512)    # memory_state: (batch_size, hidden_size)

mog_lstm = MogrifierLSTM(512,512,5)
h,c = mog_lstm(xt, (ht, ct))
print(h.shape)
print(c.shape)
```

```
(5,512)
(5,512)
```


