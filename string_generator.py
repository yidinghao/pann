import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Type

from tqdm import tqdm

from _utils import timer


class StringGenerator(ABC):
    def __init__(self, *alphabet: str, sep: str = "", colsep: str = ","):
        super(StringGenerator, self)
        self.alphabet = alphabet
        self.sep = sep
        self.colsep = colsep

    def to_string(self, indices: List[int]):
        return self.sep.join(self.alphabet[i] for i in indices)

    @abstractmethod
    def classify(self, string: str) -> bool:
        raise NotImplementedError()

    def generate(self, min_len: int, max_len: int, num_samples: int,
                 filename: str):
        n = math.ceil(num_samples / (2 * (max_len - min_len)))
        num_samples_total = n * 2 * (max_len - min_len)
        file = open(filename, "w")
        pbar = tqdm(total=num_samples_total)

        with timer("Generating {} strings...".format(num_samples_total)):
            for l_ in range(min_len, max_len):
                num_accept = 0
                num_reject = 0
                while num_accept < n and num_reject < n:
                    string = self.sep.join(random.choices(self.alphabet, k=l_))
                    accept = self.classify(string)
                    if (accept and num_accept < n) or \
                            (not accept and num_reject < n):
                        file.write("{}{}{}\n".format(string, self.colsep,
                                                     accept))
                        num_accept += accept
                        num_reject += not accept
                        pbar.update()

        pbar.close()
        file.close()


class EAGenerator(StringGenerator):
    def __init__(self):
        super(EAGenerator, self).__init__("a", "b", "c", "d")

    def classify(self, string: str) -> bool:
        return string.count("a") % 2 == 0


class SL2Generator(StringGenerator):
    def __init__(self):
        super(SL2Generator, self).__init__("a", "b", "c", "d")

    def classify(self, string: str) -> bool:
        return not (string.startswith("b") or string.endswith("a") or "aa"
                    in string or "bb" in string)


class MemorizationGenerator(StringGenerator):
    def __init__(self):
        super(MemorizationGenerator, self).__init__("a", "b", "c", "d")
        self.memory = defaultdict(lambda: bool(random.getrandbits(1)))

    def classify(self, string: str) -> bool:
        return self.memory[string]


class NoisyGenerator(StringGenerator):
    def __init__(self, cls: Type[StringGenerator], prob_flip: float, *args,
                 **kwargs):
        super(cls, self).__init__(*args, **kwargs)
        self.cls = cls
        self.prob_flip = prob_flip

    def classify(self, string: str) -> bool:
        return super(self.cls, self).classify(string) ^ \
               (random.random() < self.prob_flip)


if __name__ == "__main__":
    ea = EAGenerator()
    ea.generate(6, 26, 1000, "data/ea_train.txt")
    ea.generate(26, 51, 1000, "data/ea_test.txt")

    sl2 = SL2Generator()
    sl2.generate(6, 26, 1000, "data/sl2_train.txt")
    sl2.generate(26, 51, 1000, "data/sl2_test.txt")

    memorization = MemorizationGenerator()
    memorization.generate(6, 26, 1000, "data/memorization_train.txt")
    memorization.generate(26, 51, 1000, "data/memorization_test.txt")
