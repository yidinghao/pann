from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PANNCell(nn.RNNCellBase):
    """
    Probabilistic automaton neural network
    """

    def __init__(self, alphabet_size: int, num_states: int):
        super(PANNCell, self).__init__(0, num_states, False, alphabet_size)
        self.alphabet_size = alphabet_size
        self.num_states = num_states
        self.weight = self.weight_hh.view(alphabet_size, num_states,
                                          num_states)

    def forward(self, x: torch.LongTensor,
                h: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        """
        Reads an alphabet symbol and transitions to the next state

        :param x: A batch of alphabet symbols [batch_size,]
        :param h: The previous states [batch_size, num_states]
        :return: The next state [batch_size, num_states]
        """
        if h is None:
            h = F.one_hot(torch.zeros(len(x), dtype=torch.long),
                          num_classes=self.num_states).float()
        return (self.weight[x] @ h.unsqueeze(-1)).squeeze(-1)


class PANNAcceptor(nn.Module):
    def __init__(self, alphabet_size: int, num_states: int):
        super(PANNAcceptor, self).__init__()
        self.num_states = num_states
        self.pann = PANNCell(alphabet_size, num_states)
        self.linear = nn.Linear(num_states, 2)

    def forward(self, xs: torch.LongTensor, lengths: torch.LongTensor) -> \
            torch.FloatTensor:
        """
        Classifies a batch of strings.

        :param xs: A batch of strings [batch_size, max_length]
        :param lengths: The length of each string in the batch
            [batch_size,]
        :return: The predictions, where 1 means accept and 0 means
            reject [batch_size, 2]
        """
        max_len = min(xs.size(-1), max(lengths))
        hs = torch.empty(len(xs), self.num_states, max_len)
        h = None
        for i in range(max_len):
            h = self.pann(xs[:, i], h=h)
            hs[:, :, i] = h

        return self.linear(hs[range(len(hs)), :, lengths - 1])


if __name__ == "__main__":
    model = PANNAcceptor(3, 5)
    xs_ = torch.randint(3, (7, 6))
    lengths_ = torch.randint(6, (7,))
    print(model(xs_, lengths_))
