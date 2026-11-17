"""Display-only baseline closure for already-qualified sparse MS1 profiles."""
import numpy as np


def zero_baseline_curve(mz, intensity, connections):
    """Keep every measurement and bridge detected gaps along y=0.

    Duplicate the boundary m/z for a vertical drop/rise rather than inventing
    a sample spacing or peak width. Close the two outer edges too. Existing
    zeros need no extra vertex; unsupported/ordinary arrays pass through.
    Work and storage scale with retained points, never the omitted mass grid.
    These drawing arrays must not enter the measurement cache or calculations.
    """
    if connections is None or not mz.size:
        return mz, intensity
    gaps = np.flatnonzero(~connections)
    edges = np.empty(2 * gaps.size + 2, dtype=np.intp)
    edges[0], edges[-1] = 0, mz.size - 1
    edges[1:-1:2], edges[2:-1:2] = gaps, gaps + 1
    insert_at = edges.copy()
    insert_at[1::2] += 1  # End edges close after their observed sample.
    nonzero = intensity[edges] != 0
    if not np.any(nonzero):
        return mz, intensity
    return (np.insert(mz, insert_at[nonzero], mz[edges[nonzero]]),
            np.insert(intensity, insert_at[nonzero], 0))
