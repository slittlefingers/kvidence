"""Measured-center Gauss--Newton proposals in an orthogonal probe subspace.

Only candidate generation: the real backend must measure every proposed input.
Coordinates have per-coordinate RMS units, independent of representation width.
"""
import numpy as np


def orthogonal_directions(rng, count, width):
    if not 1 <= count <= width:
        raise ValueError('Require 1 <= count <= width')
    basis = np.linalg.qr(rng.standard_normal((width, count)), mode='reduced')[0]
    return (np.sqrt(width) * basis.T).astype(np.float32)


def centered_steps(delta, values, center_kv, target, c, multipliers=(.25,.5,1.,2.)):
    """Fit paired secants, anchor at actual center, solve damped normal equations.

    delta rows must be orthogonal with RMS one. Each proposed displacement
    is restricted to the probe radius before applying one of four step scales.
    No midpoint KV is substituted for the measured center on a nonlinear map.
    """
    delta = np.asarray(delta, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    center_kv = np.asarray(center_kv, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not len(multipliers) or not all(np.isfinite(a) and a>0 for a in multipliers):
        raise ValueError('Expected finite positive step multipliers')
    if delta.ndim != 2 or c <= 0 or not np.isfinite(c):
        raise ValueError('Invalid directions or radius')
    n, width = delta.shape
    if (n < 1 or width < 1 or target.ndim != 1 or center_kv.shape != target.shape
            or values.shape != (2*n, len(target))
            or not all(np.isfinite(a).all() for a in (delta, values, center_kv, target))):
        raise ValueError('Expected aligned finite probes, center and target')
    input_gram = np.einsum('ik,jk->ij', delta, delta, optimize=False) / width
    if not np.allclose(input_gram, np.eye(n), atol=1e-5):
        raise ValueError('Probe directions must be RMS-normalized and orthogonal')
    jacobian = (values[:n] - values[n:]) / (2*c)
    gram = np.einsum('ik,jk->ij', jacobian, jacobian, optimize=False)
    rhs = np.einsum('ik,k->i', jacobian, target-center_kv, optimize=False)
    eigen, vectors = np.linalg.eigh(gram)
    eigen = np.maximum(eigen, 0.)
    projected_rhs = np.einsum('ji,j->i', vectors, rhs, optimize=False)
    scale = max(float(np.mean(eigen)), 1e-8)
    steps = []
    for damping in (.001, .01, .1, 1.):
        weights = np.einsum('ij,j->i', vectors,
            projected_rhs / (eigen + damping*scale), optimize=False)
        # Orthogonality makes ||weights|| equal to displacement RMS.
        norm = float(np.linalg.norm(weights))
        if norm > c:
            weights *= c/norm
        direction = np.einsum('i,ij->j', weights, delta, optimize=False)
        steps.extend(direction*a for a in multipliers)
    result = np.asarray(steps, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite geometry proposal')
    return result
