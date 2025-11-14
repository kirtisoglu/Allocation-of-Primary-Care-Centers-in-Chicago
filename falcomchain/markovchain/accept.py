import random

from gerrychain.partition import Partition


def always_accept(partition: Partition) -> bool:
    return True
