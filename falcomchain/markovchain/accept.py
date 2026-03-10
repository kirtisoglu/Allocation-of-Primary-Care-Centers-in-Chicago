from falcomchain.partition import Partition


def always_accept(partition: Partition) -> bool:
    """
    Acceptance function that accepts every proposed partition unconditionally.
    Use this to run the Markov chain as a sampler without any rejection step.

    :param partition: The proposed partition.
    :type partition: Partition
    :returns: Always ``True``.
    :rtype: bool
    """
    return True
