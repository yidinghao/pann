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
        :return: The next state
        """
        if h is None:
            h = F.one_hot(torch.zeros(len(x), dtype=torch.long),
                          num_classes=self.alphabet_size)
        return (self.weight[x] @ h.unsqueeze(-1)).squeeze(-1)
