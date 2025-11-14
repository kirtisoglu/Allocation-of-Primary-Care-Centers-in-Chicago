class BipartitionWarning(UserWarning):
    """
    Generally raised when it is proving difficult to find a balanced cut.
    """

    pass


class ReselectException(Exception):
    """
    Raised when the tree-splitting algorithm is unable to find a
    balanced cut after some maximum number of attempts, but the
    user has allowed the algorithm to reselect the pair of
    districts from parent graph to try and recombine.
    """

    pass


class BalanceError(Exception):
    """Raised when a balanced cut cannot be found."""


class PopulationBalanceError(Exception):
    """Raised when the population of a district is outside the acceptable epsilon range."""
