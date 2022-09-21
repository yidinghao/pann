from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

Alphabet = Union[List[str], Tuple[str]]


def str_to_idx(string: str, alphabet: Alphabet = ("a", "b", "c", "d"),
               sep: str = "") -> List[int]:
    return [alphabet.index(x)
            for x in (string.split(sep) if len(sep) > 0 else string)]


def load_data(filename: str, alphabet: Alphabet = ("a", "b", "c", "d"),
              batch_size: Optional[int] = None, colsep: str = ",",
              num_batches: int = 100000, sep: str = ""):
    if batch_size is None:
        raise ValueError("batch_size can't be None!")
    alphabet = list(alphabet)

    # Load data into tensors
    with open(filename, "r") as f:
        all_data = [line.strip().split(colsep) for line in f]

    print(len(all_data))
    all_xs = [str_to_idx(x, alphabet=alphabet, sep=sep) for x, _ in all_data]
    all_ys = [int(y == "True") for _, y in all_data]
    del all_data

    # Split into batches
    pointer = 0
    for i in range(num_batches):
        print(i, pointer, sep=":")
        xs = all_xs[pointer:pointer + batch_size]
        ys = all_ys[pointer:pointer + batch_size]
        pointer += batch_size
        if pointer >= len(all_xs):
            pointer -= len(all_xs)
            xs += all_xs[:pointer]
            ys += all_ys[:pointer]

        lengths = torch.LongTensor([len(x) for x in xs])
        xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True)
        ys = torch.LongTensor(ys)

        yield xs, ys, lengths


if __name__ == "__main__":
    i = 0
    for xs_, ys_, lengths_ in load_data("data/ea_train.txt", batch_size=7):
        i += 1
        if i == 143:
            print(xs_)
            print(ys_)
            print(lengths_)
            quit()
