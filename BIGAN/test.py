import os
from BIGAN.bigan import BIGAN


if __name__ == '__main__':
    iterations = 100
    input_dim = 64
    NUMBER_OF_TESTS = 120

    bigan = BIGAN()
    bigan.load_weights()



