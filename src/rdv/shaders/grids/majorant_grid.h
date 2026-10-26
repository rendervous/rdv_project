/*
Code for the MajorantGrid class. Each output voxel is the elementwise
maximum over every fine voxel that a continuous-coordinate sampler (a
GridSampler/TensorSampler/GPUSampler - NOT `grid` itself, which is only
ever indexed here, never interpolated) reading `grid` under the *same*
aligned/linear convention as `aligned`/`linear` below could possibly read
from anywhere inside this coarse cell's own spatial extent:

- `aligned` changes the continuous-coordinate -> fine-voxel-position
  formula, exactly like GridSampler/TensorSampler/GPUSampler's own
  `aligned` (see shaders/samplers/grid_sampler.h) - aligned=True ties
  -1/+1 to the outer voxel centers, aligned=False to the volume's outer
  edges - so the SAME coarse cell maps to a different fine-voxel range
  depending on it.
- `linear` (bilinear/trilinear/... interpolation) means a query anywhere
  inside this cell can blend in the fine voxel one step *outside* the
  cell's own naive index range (the other end of whichever corner pair
  straddles the cell boundary) - excluding it would let the true
  (interpolated) density exceed this "majorant" right at the border,
  silently biasing delta tracking (this is the abrupt-density-at-the-
  volume-boundary case in particular: a coarse cell touching the outer
  face of the box must also cover the one fine voxel linear filtering
  blends in from just past that face). `linear=False` (nearest) only ever
  reads one (rounded) fine voxel per query - its true, tighter reach is
  used instead, letting the tracker skip more empty space.

Either way, the result is still a plain piecewise-constant grid (one fixed
float per coarse cell) - `aligned`/`linear` only change *which* fine
voxels get folded into each cell's own max during construction, not the
shape of MajorantGrid's own output.

Passing `aligned`/`linear` that don't match whatever sampler is actually
built on top of `grid` at render time produces an invalid (possibly biased)
bound - they must be the same values that sampler was itself constructed
with.
*/

void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    int fine_shape[IN];
    for (int d = 0; d < IN; d++)
        fine_shape[d] = deref(deref(node).grid).shape[d];

    bool aligned = deref(node).aligned != 0;
    bool lin = deref(node).linear != 0;

    int lo[IN];
    int extent[IN];
    int count = 1;
    for (int d = 0; d < IN; d++) {
        int coarse_n = deref(node).shape[d];
        float fsz = float(fine_shape[d]);
        // Fractional (0..1) span of this coarse cell along dimension d,
        // converted to the same continuous fine-voxel "position" space
        // GridSampler/TensorSampler/GPUSampler use (see their own
        // aligned formula) - monotonic in t, so the cell's own pos range
        // is exactly [pos(t_lo), pos(t_hi)].
        float t_lo = float(_input[d]) / float(coarse_n);
        float t_hi = float(_input[d] + 1) / float(coarse_n);
        float pos_lo = aligned ? t_lo * (fsz - 1.0) : t_lo * fsz - 0.5;
        float pos_hi = aligned ? t_hi * (fsz - 1.0) : t_hi * fsz - 0.5;

        int d_lo, d_hi; // fine voxel index range [d_lo, d_hi) covering every voxel reachable from this cell
        if (lin) {
            // linear reads floor(pos) and floor(pos)+1 - the +1 corner at
            // the high end can reach one voxel past floor(pos_hi).
            d_lo = int(floor(pos_lo));
            d_hi = int(floor(pos_hi)) + 2;
        } else {
            // nearest reads round(pos) = floor(pos + 0.5), monotonic in pos.
            d_lo = int(floor(pos_lo + 0.5));
            d_hi = int(floor(pos_hi + 0.5)) + 1;
        }
        if (d_lo < 0)
            d_lo = 0;
        if (d_hi > fine_shape[d])
            d_hi = fine_shape[d];
        if (d_hi <= d_lo)
            d_hi = d_lo + 1;
        if (d_hi > fine_shape[d])
            d_hi = fine_shape[d];
        if (d_lo >= d_hi)
            d_lo = d_hi - 1;
        lo[d] = d_lo;
        extent[d] = d_hi - d_lo;
        count *= extent[d];
    }

    int idx[IN];
    for (int c = 0; c < count; c++) {
        int rem = c;
        for (int d = IN - 1; d >= 0; d--) {
            idx[d] = lo[d] + (rem % extent[d]);
            rem /= extent[d];
        }
        float values[OUT];
        forward(deref(node).grid, idx, values);
        if (c == 0) {
            _output = values;
        } else {
            for (int k = 0; k < OUT; k++)
                _output[k] = max(_output[k], values[k]);
        }
    }
}
