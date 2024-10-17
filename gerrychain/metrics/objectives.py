import math
from typing import Dict
import numpy as np

def squared_radius_deviation(partition):
    """
    Calculate the sum of squared deviations from the mean for radius values.
    
    :param p: Partition Partition class with a 'radius' attribute 
    :return: The sum of squared deviations from the mean
    """
    radius_array = np.array(list(partition.radius.values()))
    mean_radius = np.mean(radius_array)  # mean
    deviations = radius_array - mean_radius # deviations from mean
    squared_deviations = deviations ** 2 # # Square deviations
    sum_squared_deviations = np.sum(squared_deviations) # Sum of squared deviations
    return sum_squared_deviations

def total_cut_edges(partition):
    return len(partition["cut_edges"])


def  arcesse_compactness(partition):
    
    return


def compute_polsby_popper(area: float, perimeter: float) -> float:
    """
    Computes the Polsby-Popper score for a single district.

    :param area: The area of the district
    :type area: float
    :param perimeter: The perimeter of the district
    :type perimeter: float

    :returns: The Polsby-Popper score for the district
    :rtype: float
    """
    try:
        return 4 * math.pi * area / perimeter**2
    except ZeroDivisionError:
        return math.nan


# Partition type hint left out due to circular import
# def polsby_popper(partition: Partition) -> Dict[int, float]:
def polsby_popper(partition) -> Dict[int, float]:
    """
    Computes Polsby-Popper compactness scores for each district in the partition.

    :param partition: The partition to compute scores for
    :type partition: Partition

    :returns: A dictionary mapping each district ID to its Polsby-Popper score
    :rtype: Dict[int, float]
    """
    return {
        part: compute_polsby_popper(
            partition["area"][part], partition["perimeter"][part]
        )
        for part in partition.parts
    }
