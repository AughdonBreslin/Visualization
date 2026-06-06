# Isomap walkthrough: timestamped companion

This text accompanies the Isomap animation. It is meant to be read alongside the
video or on its own. The video runs about 2 minutes 43 seconds and is divided into
six steps, matching the pseudocode panel shown in the top-left corner throughout.
Each entry below gives a timestamp, the caption that appears on screen at that
moment, and a fuller explanation than the caption has room for. Timestamps mark
when a caption begins to appear; it stays on screen until the next one.

Notation used throughout: there are `n` data points. If we knew their coordinates,
`X` is the `n` by `d` matrix whose row `i` is point `x_i`. `D` is the `n` by `n`
matrix of pairwise distances. `B` is the matrix produced in step 4. The full
algebra of step 4 is also written out in `double_centering_explained.md`.

## What Isomap does, in one paragraph

Ordinary linear projection (PCA) flattens data by ignoring its curvature, so it
tears or folds a curved sheet. Isomap instead measures distance along the sheet,
not straight through the surrounding space, and then finds the flat arrangement of
points that best reproduces those along-the-sheet distances. The six steps are:
build a neighbor graph, measure geodesic (graph) distances, convert that distance
table into a matrix of inner products by double-centering, factor that matrix with
an eigendecomposition, and read the flat coordinates off the top eigenvectors.

## Step 1: a curved sheet in space (0:00 to 0:13)

The scene opens on a Swiss roll: a flat two-dimensional sheet that has been rolled
up inside three-dimensional space. Faint x, y, z axes sit behind it, and the cloud
rotates slowly so the rolled structure is visible from several angles. Color varies
along the sheet so you can track which parts are near each other on the surface.

[0:03] "A 2D sheet rolled up in 3D. The goal: recover the flat sheet."

The point of the whole method is stated up front. The data lives in 3D, but it only
has two genuine degrees of freedom, because it is a surface. Two points can sit close
together in 3D while being far apart along the sheet, the way two layers of a rolled
poster nearly touch but are far apart along the paper. Recovering the flat sheet means
finding a 2D coordinate for every point that respects distance along the surface.

## Step 2: the neighbor graph (0:13 to 0:33)

The full cloud fades out and a small flat diagram takes its place, so the rule for
building the graph can be seen clearly without the clutter of a thousand points.

[0:16] "To build the graph, link each point to its nearest neighbors."

A central point appears, then six nearby points around it, each joined to the center
by an edge. This is the k-nearest-neighbor rule: every point is connected to its `k`
closest points (the animation uses `k = 8` for the real data). The graph is the
scaffold that will let us measure distance along the sheet, because short hops between
near neighbors stay on the surface.

[0:21] "Each link is weighted by the distance between the points."

The central point and its nearest neighbors appear first, under the caption above, so
the neighborhood is on screen before weighting is introduced. Then each link is drawn in
turn with its weight at the midpoint.

As each edge is drawn, a number appears at its midpoint: the straight-line distance
between the two endpoints. These weights matter. A neighbor edge is only ever between
points that are genuinely close, so its straight-line length is a good estimate of the
true surface distance over that short span. The weights are what later get added up
along paths.

[0:28] "Do this for every point and the graph lights up across the data."

The schematic clears, the real cloud returns, and the full neighbor graph sweeps into
view as thousands of short edges. Each point contributed its own local star of links;
together they form a mesh that follows the sheet. The cloud then begins a continuous
slow rotation that carries through into the next step.

## Step 3: geodesic distance (0:33 to 0:55)

Definition (stated here rather than in the video): a geodesic distance is the length
of the shortest route between two points that stays on the surface. On the neighbor
graph this is the shortest path between two nodes, found by adding up edge weights. It
is the graph stand-in for true surface distance, and it is what separates Isomap from a
method that would just use straight-line distance.

The step opens with a wavefront spreading out from one source point across the graph.
This is Dijkstra's algorithm: it settles nodes in order of increasing distance from the
source, always extending the closest frontier first, which is exactly how the
shortest-path distances are computed.

[0:34] "Color shows geodesic distance from the source point."

The whole cloud recolors by graph distance from the source, using a high-saturation
rainbow that spans from blue at the near end through cyan, green, yellow, and orange to
red at the far end. Notice that two points which are close in 3D but on different turns
of the roll get very different colors, because the path between them has to travel all
the way around. The same rainbow is reused for the final 2D embedding in step 6, so the
preserved gradient is easy to recognize.

[0:46] "Straight-line distance cuts through space, off the sheet."

A straight segment connects a source and a far target, cutting directly through the
empty space between the rolls. This is the quantity Isomap deliberately does not use.
It is short in 3D but meaningless for the sheet, because it leaves the surface.

[0:50] "Geodesic distance follows the graph along the sheet."

A second path is drawn, this one hugging the surface along graph edges from source to
target. It is longer than the straight segment, and it is the honest measure of how far
apart the two points are on the sheet. The cloud keeps rotating during both paths so
the difference in depth between the chord and the surface route is visible, and it keeps
rotating as it fades out into the next step rather than freezing first.

## Step 4: from distances to inner products (0:55 to 2:00)

This is the mathematical core. We have a table of geodesic distances. We want actual
coordinates. The step converts one into the other. The full derivation is in
`double_centering_explained.md`; what follows explains each on-screen line.

[1:00] "We have geodesic distances. We want coordinates: a position for each point."

Four sample points are shown above an empty four-by-four grid. The grid will hold the
distances among those four points, and is a stand-in for the full `n` by `n` table. The
four points are chosen spread out across the sheet (by farthest-point sampling) rather
than along one line, so that the small Gram matrix built from them later in step 5 is
non-degenerate and has a genuine second eigenvalue to keep.

[1:03] "Each entry is the distance between two points; the diagonal is zero."

The grid fills one cell at a time. The diagonal fills first: the distance from a point
to itself is zero. Then each off-diagonal cell lights up the two points it refers to
and draws the link between them, so it is clear that the number in row `i`, column `j`
is the geodesic distance between point `i` and point `j`. The table is symmetric,
because the distance from `i` to `j` equals the distance from `j` to `i`.

[1:12] "Since rotating or shifting all the points together would yield identical pairwise distances,"

Here is the obstacle, stated as the first half of a single thought that the next caption
completes. Take any arrangement of points and move the whole thing as a rigid body: slide
it, spin it, or flip it. Every pairwise distance is unchanged, because rigid motions
preserve lengths.

[1:16] "distances alone could not create unique coordinates, since many placements share one distance table."

Because of that invariance, the map from coordinates to distances is many-to-one. Reading
distances backward can only ever recover the shape, never a single placement with a fixed
position and orientation. We will need an extra convention (fixing the center at the origin)
to proceed.

[1:22] "Instead, we can base it off of the inner product of x_i and x_j and the angle between them from a shared origin."

The inner product (dot product) of two position vectors equals the product of their
lengths times the cosine of the angle between them. A single number therefore carries
both lengths and the angle, all measured from a common origin. Unlike a distance, it
depends on where the origin sits, which is exactly the sensitivity needed to pin down
coordinates.

[1:27] "The Gram matrix G is the result of collecting every inner product, equal to X times X-transpose."

Stacking all the inner products `G_ij = x_i . x_j` into one `n` by `n` matrix gives the
Gram matrix. It has the compact form `G = X X-transpose`, so the Gram matrix is a single
matrix product away from the coordinates.

[1:31] "G captures the geometric relationships from relative geometry, a bridge between distances and coordinates."

`G` records every length and every angle, the relative geometry of the configuration,
expressed against one shared origin. That makes it the bridge we need: distances can be
converted into `G` (the next three lines do this), and `G` can be factored back into
coordinates. It sits squarely between the thing we can measure and the thing we want.

[1:37] "By eigendecomposing the Gram matrix, we find the eigenvectors that best explain the data's geometry."

`G` is symmetric and positive semidefinite, so the spectral theorem factors it as
`G = V Λ V-transpose`, with orthonormal eigenvectors in `V` and eigenvalues in `Λ` (capital
lambda). The eigenvectors with the largest eigenvalues are the directions in which the
points vary the most, the genuine low-dimensional structure. Scaling each kept eigenvector
by the square root of its eigenvalue rebuilds the coordinates, `X = V √Λ`; the worked
algebra is in `double_centering_explained.md`. This previews step 5, which performs the
eigendecomposition.

[1:41] "We cannot form G from coordinates we do not have, so we build it from the distances instead."

The three lines that follow build `G` from the distance table, because we do not have the
coordinates `X` needed to compute `G = X X-transpose` directly. A squared distance is a
fixed combination of inner products, and inverting that relation by squaring and then
double-centering produces `G` (called `B` once centered).

[1:45] "First, square every entry of the distance matrix."

The distance grid transforms into the grid of squared distances (each number squared). This
is the `D squared` matrix that the double-centering formula acts on.

[1:51] "Then double-center it: subtract each row and column mean, and scale by minus one half."

Applying the operation `B = -1/2 J D-squared J`, where `J` is the centering matrix, subtracts
each row mean and each column mean from the squared-distance matrix and scales the result by
minus one half. Term by term, the `|x_i| squared` and `|x_j| squared` pieces are constant along
rows and columns, so centering removes them, and the surviving `-2 (x_i . x_j)` piece becomes
the inner product once the minus one half cancels the minus two. The centering is also the
convention from earlier that fixes the origin at the centroid of the points.

[1:56] "What remains is B, the Gram matrix of inner products, ready to factor into coordinates."

What remains, `B`, is the Gram matrix of the centered points: the inner products we wanted,
computed from distances alone. It is exactly the kind of matrix the eigendecomposition
applies to, which is what the next step does.

## Step 5: eigendecomposition (2:00 to 2:35)

This step factors the Gram matrix `B`. The whole step is shown on the visible four-by-four
sample matrix, with real vectors and real numbers, so the multiplication can be watched
directly. The reported numbers below are for the 1000-point dataset; the four-by-four sample
has eigenvalues about 219.0 and 14.6.

[2:02] "Power iteration starts from a random unit vector v."

Power iteration is the simplest way to find the dominant eigenvector. It needs a starting
vector, and that vector is just a random one, rescaled to length 1. The animation shows that
starting vector `v_0` as a column of four numbers next to `B`, labeled as random with norm 1,
so it is clear the method is not given the answer; it begins from noise.

[2:07] "Each step multiplies v by B, then divides by its length."

The update rule shown on top is `v <- B v / norm(B v)`. Multiplying by `B` stretches the
vector most along the direction of the largest eigenvalue. Dividing by the length keeps the
vector at norm 1 so it does not blow up. Repeating this rule is the entire algorithm, and it
is how each iteration's vector is produced from the previous one.

[2:10] "The value is v-transpose times B times v: an actual matrix product."

The quantity being tracked is the Rayleigh quotient `v-transpose B v`. The animation lays it
out as a literal matrix product: the row vector `v-transpose` on the left, the four-by-four
matrix `B` in the middle, the column vector `v` on the right, and the resulting single number
to the side. For a unit vector this number is a weighted average of the eigenvalues, weighted
by how much of `v` points along each eigenvector.

[2:15] "Multiply by B and renormalize; the value climbs each iteration."

Each iteration updates the row and column vectors and recomputes the product. The value climbs
across the iterations (for the 1000-point sample it runs about 8.1, then 77.1, then 217.0, then
219.0) because `v` is turning toward the top eigenvector, and the Rayleigh quotient is largest
exactly when `v` is that eigenvector.

[2:23] "The value settles at the top eigenvalue lambda-1."

Once `v` has aligned with the top eigenvector, the product stops changing: it has reached the
largest eigenvalue, lambda one (about 219.0 here). That is the defining property used in the
next line: at an eigenvector, `B v` is just a scalar times `v`, and that scalar is the
eigenvalue.

[2:29] "Every eigenvector satisfies B v = λ v; the largest eigenvalues carry the shape."

The eigenvalue equation `B v = λ v` says applying `B` to an eigenvector only stretches it,
never rotates it. The eigenvectors with the largest eigenvalues are the directions in which the
centered points vary the most, which is where the genuine two-dimensional structure of the
unrolled sheet lives. Small eigenvalues correspond to noise or to dimensions the flat sheet
does not use.

[2:33] "Keep the top two eigenvalues: they span the recovered plane."

The two largest eigenvalues, lambda one and lambda two (about 219.0 and 14.6 for the sample),
are kept, and their eigenvectors span the plane the sheet will be laid out in. Everything else
is discarded, which is the reduction from three dimensions (or more) down to two.

## Step 6: the flat embedding (2:35 to 2:43)

[2:38] "The sheet unrolls into 2D. Color shows geodesic distance from the source, preserved."

The final coordinates are `Y = [root(lambda1) v1, root(lambda2) v2]`, the formula shown at the
top right. This is the same operation as step 3, scaling each kept eigenvector by the square root
of its eigenvalue, applied to the top two components only. Point `i` lands at
`(root(lambda1) v1[i], root(lambda2) v2[i])`. The cloud settles into a flat rectangle: the sheet
unrolled. The coloring is the same geodesic-distance coloring from step 3, and it now varies
smoothly across the flat layout, which is the visual confirmation that along-the-sheet distances
were preserved while the extra dimension was removed.
