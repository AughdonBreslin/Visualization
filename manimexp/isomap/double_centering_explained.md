# Double centering and classical MDS: the math behind step 4

This document expands the captions in section 4 of the Isomap walkthrough (the
double-centering step). The video states the ideas in short beats; here each one
is worked out in full. Notation: there are `n` points. If we knew their
coordinates, `X` is the `n by d` matrix whose row `i` is the point `x_i` (a vector
in `d`-dimensional space). `D` is the `n by n` matrix of pairwise distances,
`D_ij = |x_i - x_j|`. `B` is the matrix the step produces.

## 1. Why distances cannot fix coordinates

Caption: "Rotate or shift all the points together and every pairwise distance
stays the same. So distances can't pin down coordinates by themselves."

Take any configuration of points and move the whole thing as a rigid body. There
are three kinds of motion that leave all pairwise distances unchanged:

- Translation: replace every `x_i` by `x_i + t` for a fixed vector `t`. Then
  `(x_i + t) - (x_j + t) = x_i - x_j`, so the distance is identical.
- Rotation: replace every `x_i` by `R x_i` for a rotation matrix `R`. Rotations
  preserve length, so `|R x_i - R x_j| = |R(x_i - x_j)| = |x_i - x_j|`.
- Reflection: same argument as rotation, with `R` a reflection.

So if one configuration has a given distance matrix, then every rotated,
reflected, and shifted copy of it has the exact same distance matrix. The map
from coordinates to distances is many-to-one. Inverting it can only recover the
configuration up to those motions, never a single unique set of coordinates.

This is not a defect for our purpose. We want the shape of the unrolled sheet,
and shape is exactly what survives rotation, reflection, and translation. The
practical fix is to remove the translation freedom by a convention: place the
centroid of the points at the origin. That is what centering does, and it is the
reason the final formula has the centering matrix `J` in it.

## 2. Inner products, and why they are the natural quantity

Caption: "An inner product x_i . x_j records the two points' lengths and the
angle between them, from a shared origin. Collect every inner product into one
matrix: the Gram matrix G, equal to X times X-transpose."

The inner product (dot product) of two position vectors is

    x_i . x_j = sum over k of x_i[k] * x_j[k] = |x_i| * |x_j| * cos(theta_ij)

where `theta_ij` is the angle between the two vectors and `|x_i|` is the length
of `x_i`. So a single number carries both lengths and the angle between the
points, all measured from the common origin. Three reasons this is the natural
quantity to work with:

1. It is bilinear in the coordinates. Stacking the numbers `G_ij = x_i . x_j` into
   an `n by n` matrix gives exactly

       G = X X-transpose

   so the Gram matrix is one matrix product away from the coordinates. "Stacking"
   here just means arranging the `n squared` inner products into a grid indexed by
   row `i` and column `j`.

2. Distances are built from inner products, not the other way around:

       |x_i - x_j| squared = (x_i - x_j) . (x_i - x_j)
                           = x_i . x_i + x_j . x_j - 2 * (x_i . x_j)
                           = G_ii + G_jj - 2 * G_ij.

   So inner products are the more fundamental object; the distance matrix is a
   function of the Gram matrix.

3. Inner products are not translation invariant. Shift the origin and every
   `x_i . x_j` changes. That sensitivity is exactly why inner products can pin
   down coordinates once we fix the origin, while distances (section 1) cannot.

## 3. How a Gram matrix factors, and how that rebuilds the coordinates

Caption: "G is symmetric, so it splits into orthogonal eigenvectors V and
eigenvalues Λ. Scale each eigenvector by the square root of its eigenvalue and
the coordinates come back: X = V√Λ." (Λ is capital lambda, the diagonal matrix of
eigenvalues; √Λ is its entrywise square root.)

Two properties of `G = X X-transpose`:

- It is symmetric: `G-transpose = (X X-transpose)-transpose = X X-transpose = G`.
- It is positive semidefinite: for any vector `v`,
  `v-transpose G v = v-transpose X X-transpose v = |X-transpose v| squared >= 0`.
  So none of its eigenvalues are negative.

The spectral theorem says any real symmetric matrix can be written

    G = V Λ V-transpose

where `V` is orthogonal (its columns `v_1, ..., v_n` are orthonormal eigenvectors)
and `Λ = diag(lambda_1, ..., lambda_n)` holds the eigenvalues. Because `G` is
positive semidefinite, every `lambda_k >= 0`, so each eigenvalue has a real square
root and we can split `Λ = √Λ √Λ` with `√Λ = diag(sqrt(lambda_1),
..., sqrt(lambda_n))`. Then

    G = V Λ V-transpose
      = V √Λ √Λ V-transpose
      = (V √Λ)(V √Λ)-transpose.

Compare this with `G = X X-transpose`. Reading off the factor gives

    X = V √Λ.

Concretely, coordinate axis `k` is the eigenvector `v_k` (a length-`n` vector, one
entry per point) scaled by `sqrt(lambda_k)`. Point `i` receives coordinates

    x_i = ( sqrt(lambda_1) * v_1[i], sqrt(lambda_2) * v_2[i], ... ).

This `X` reproduces `G` exactly, hence every inner product, hence every distance.
It is determined only up to an orthogonal transformation, which matches section 1:
distances and inner products fix the shape, not the orientation.

The dimension reduction is the payoff. The rank of `G` equals the rank of `X`,
which is the true dimension `d`. The other eigenvalues are zero (or near zero with
noisy data), so keeping the top `d` eigenpairs loses nothing essential. Isomap
keeps the top two:

    Y = [ sqrt(lambda_1) * v_1, sqrt(lambda_2) * v_2 ].

That is the formula in step 6, and it is the same operation as "factor the Gram
matrix and read off coordinates," applied to the top two components only.

## 4. Why -1/2 J D-squared J is the Gram matrix

Caption: "A squared distance is just a combination of inner products, so the
distance table converts straight back into G."

We do not actually have `X`, so we cannot compute `G = X X-transpose` directly. But
section 2 gave the link between squared distances and inner products:

    D_ij squared = G_ii + G_jj - 2 * G_ij.

The goal is to invert this and recover `G` from `D-squared` alone. The obstacle is
the `G_ii` and `G_jj` terms. The fix is centering. Let `J` be the centering matrix

    J = I - (1/n) * 1 1-transpose

where `1` is the all-ones vector. Multiplying a matrix by `J` on the left subtracts
each column's mean; multiplying by `J` on the right subtracts each row's mean.
Applying `J` on both sides of `D-squared` and scaling by `-1/2`:

    B = -1/2 * J D-squared J.

Work through what centering does to each of the three terms of
`D_ij squared = G_ii + G_jj - 2 G_ij`:

- `G_ii` is constant along each row, so it sits entirely in the row means and is
  removed by the centering on that side.
- `G_jj` is constant along each column, so it is removed by the centering on the
  other side.
- The `-2 G_ij` term, once the points are centered so that the column means and
  row means of `G` are zero, passes through unchanged.

What remains, after the `-1/2` cancels the `-2`, is

    B = G_centered = X_c X_c-transpose,

the Gram matrix of the centered coordinates `X_c` (the points with their centroid
moved to the origin). So `B` is a Gram matrix of inner products that we computed
using distances only. Feeding `B` into the eigendecomposition of section 3 (steps
5 and 6 of the video) factors it and reads off the coordinates. That is the whole
arc: distances, to squared distances, to double-centered inner products `B`, to
eigenvectors, to the recovered flat coordinates.
