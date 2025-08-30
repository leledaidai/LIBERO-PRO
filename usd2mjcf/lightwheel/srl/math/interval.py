# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""SRL general utility / helper functions for axis-aligned intervals."""

# Standard Library
from itertools import product
from typing import List, Optional, Tuple

# Third Party
import numpy as np

# NVIDIA
from lightwheel.srl.basics.types import Vector

Interval = Tuple[Vector, Vector]


def get_center(interval: Interval) -> Vector:
    """Get the center of defined by two bounds.

    Example:
        ```
        >>> interval.get_center(([1, 2, 3], [3, 4, 5]))
        [2, 3, 4]
        ```
    """
    lower, upper = interval
    center = np.average((np.array(lower), np.array(upper)), axis=0)
    if isinstance(lower, np.ndarray):
        return center
    return center.tolist()


def get_extent(interval: Interval) -> Vector:
    """Get the extent of interval.

    Examples:
        ```
        >>> interval.get_extent(([1, 2, 3], [4, 6, 8]))
        [3, 4, 5]
        ```
    """
    lower, upper = interval
    extent = np.array(upper) - np.array(lower)
    if isinstance(lower, np.ndarray):
        return extent
    return extent.tolist()


def get_dimensionality(interval: Interval) -> int:
    """Get the dimensionality of the interval.

    Examples:
        ```
        >>> interval.get_dimensionality(([0, 0, 0], [0, 1, 2]))
        3
        ```
    """
    [_, dim] = np.array(interval).shape
    return dim


def is_empty(interval: Interval) -> bool:
    """Check whether an interval is empty.

    Examples:
        ```
        >>> interval.is_empty(([0, 0], [-1, 1]))
        True
        ```
    """
    lower, upper = interval
    return bool(np.less(upper, lower).any())


def get_volume(interval: Interval) -> float:
    """Get the volume of an interval.

    Examples:
        ```
        >>> interval.get_volume(([0, 0], [1, 2]))
        2
        ```
    """
    if is_empty(interval):
        return 0.0
    return float(np.prod(get_extent(interval)))


def construct_from_vectors(vectors: List[Vector]) -> Interval:
    """Construct the interval for a list of vectors.

    Examples:
        ```
        >>> interval.construct_from_vectors([[10, 1, 6], [5, 5, 5]])
        (array([5, 1, 5]), array([10, 5, 6])
        ```
    """
    lower = np.min(np.array(vectors), axis=0)
    upper = np.max(np.array(vectors), axis=0)
    if isinstance(vectors[0], np.ndarray):
        return (lower, upper)
    return (lower.tolist(), upper.tolist())


def contains_vector(interval: Interval, vector: Vector) -> bool:
    """Check whether the interval contains a vector.

    Examples:
        ```
        >>> interval.contains_vector(([1, 2, 3], [4, 5, 6]), [3, 3, 3])
        True
        ```
    """
    lower, upper = interval
    return bool(np.less_equal(lower, vector).all() and np.less_equal(vector, upper).all())


def contains_interval(interval1: Interval, interval2: Interval) -> bool:
    """Check whether interval contain another interval.

    Examples:
        ```
        >>> interval.contains_interval(([1, 2, 3], [4, 5, 6]), ([3, 3, 3], [4, 4, 4]))
        True
        ```
    """
    lower2, upper2 = interval2
    return contains_vector(interval1, lower2) and contains_vector(interval1, upper2)


def overlaps_interval(interval1: Interval, interval2: Interval) -> bool:
    """Check whether two interval overlap.

    Examples:
        ```
        >>> interval.overlaps_interval(([1, 2, 3], [4, 5, 6]), ([3, 3, 3], [7, 7, 7]))
        True
        ```
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    return bool(np.less_equal(lower1, upper2).all() and np.less_equal(lower2, upper1).all())


def union_intervals(interval1: Interval, interval2: Interval) -> Interval:
    """Compute the union of two intervals.

    Examples:
        ```
        >>> interval.union_intervals(([0, 0], [2, 2]), ([1, 1], [3, 3]))
        ([0, 0], [3, 3])
        ```
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    lower_union = np.minimum(lower1, lower2)
    upper_union = np.maximum(upper1, upper2)
    if isinstance(lower1, np.ndarray):
        return (lower_union, upper_union)
    return (lower_union.tolist(), upper_union.tolist())


def intersect_intervals(interval1: Interval, interval2: Interval) -> Optional[Interval]:
    """Compute the intersection of two intervals.

    Examples:
        ```
        >>> interval.intersect_intervals(([0, 0], [2, 2]), ([1, 1], [3, 3]))
        ([1, 1], [2, 2])
        ```
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    lower_intersection = np.maximum(lower1, lower2)
    upper_intersection = np.minimum(upper1, upper2)
    interval_intersection = (lower_intersection, upper_intersection)
    if is_empty(interval_intersection):
        return None
    if isinstance(lower1, np.ndarray):
        return interval_intersection
    return (lower_intersection.tolist(), upper_intersection.tolist())


def get_vertices(interval: Interval) -> List[np.ndarray]:
    """Get the vertices of an interval.

    Examples:
        ```
        >>> interval.get_vertices(([0, 0], [1, 2]))
        [[0, 0], [0, 2], [1, 0], [1, 2]]
        ```
    """
    dim = get_dimensionality(interval)
    vertices = [
        np.array([interval[index][coord] for coord, index in enumerate(indices)], dtype=np.float64)
        for indices in product(range(2), repeat=dim)
    ]
    if isinstance(interval[0], np.ndarray):
        return vertices
    return [vertex.tolist() for vertex in vertices]


def sample_uniformly(
    interval: Interval, random_state: Optional[np.random.RandomState] = None
) -> Vector:
    """Sample a vector uniformly at random over an interval.

    Examples:
        ```
        >>> interval.sample_uniformly(([0, 0], [100, 100]))
        [54.88135039 71.51893664]
        ```
    """
    if random_state is None:
        random_state = np.random.RandomState()
    lower, upper = interval
    vector = random_state.uniform(lower, upper)
    if isinstance(lower, np.ndarray):
        return vector
    return vector.tolist()
