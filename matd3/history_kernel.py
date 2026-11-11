"""Bounded I3 distances without allocating a feature array per candidate pair.

No fastmath or parallel reduction: reproduce NumPy's float32 eight-lane
pairwise sum for contiguous feature rows of length 8..128. Other layouts and
widths use the existing NumPy path in history_enhancement.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def eligible_distances(features, queries, rows, columns):
    width = features.shape[1]
    result = np.empty(len(rows), np.float32)
    end = width-width%8
    for pair in range(len(rows)):
        a, b = columns[pair], rows[pair]
        d0 = np.float32(features[a, 0]-queries[b, 0]); r0 = np.float32(d0*d0)
        d1 = np.float32(features[a, 1]-queries[b, 1]); r1 = np.float32(d1*d1)
        d2 = np.float32(features[a, 2]-queries[b, 2]); r2 = np.float32(d2*d2)
        d3 = np.float32(features[a, 3]-queries[b, 3]); r3 = np.float32(d3*d3)
        d4 = np.float32(features[a, 4]-queries[b, 4]); r4 = np.float32(d4*d4)
        d5 = np.float32(features[a, 5]-queries[b, 5]); r5 = np.float32(d5*d5)
        d6 = np.float32(features[a, 6]-queries[b, 6]); r6 = np.float32(d6*d6)
        d7 = np.float32(features[a, 7]-queries[b, 7]); r7 = np.float32(d7*d7)
        for start in range(8, end, 8):
            d0 = np.float32(features[a, start]-queries[b, start]); r0 = np.float32(r0+np.float32(d0*d0))
            d1 = np.float32(features[a, start+1]-queries[b, start+1]); r1 = np.float32(r1+np.float32(d1*d1))
            d2 = np.float32(features[a, start+2]-queries[b, start+2]); r2 = np.float32(r2+np.float32(d2*d2))
            d3 = np.float32(features[a, start+3]-queries[b, start+3]); r3 = np.float32(r3+np.float32(d3*d3))
            d4 = np.float32(features[a, start+4]-queries[b, start+4]); r4 = np.float32(r4+np.float32(d4*d4))
            d5 = np.float32(features[a, start+5]-queries[b, start+5]); r5 = np.float32(r5+np.float32(d5*d5))
            d6 = np.float32(features[a, start+6]-queries[b, start+6]); r6 = np.float32(r6+np.float32(d6*d6))
            d7 = np.float32(features[a, start+7]-queries[b, start+7]); r7 = np.float32(r7+np.float32(d7*d7))
        left = np.float32(np.float32(r0+r1)+np.float32(r2+r3))
        right = np.float32(np.float32(r4+r5)+np.float32(r6+r7))
        total = np.float32(left+right)
        for j in range(end, width):
            d = np.float32(features[a, j]-queries[b, j])
            total = np.float32(total+np.float32(d*d))
        result[pair] = np.sqrt(np.float32(total/width))
    return result
