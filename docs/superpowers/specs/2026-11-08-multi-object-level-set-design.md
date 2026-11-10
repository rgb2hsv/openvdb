# Multi-Object Level Set (MOGAC) for OpenVDB — Design

Date: 2026-11-08
Status: Draft for review
Target: upstream OpenVDB, header-only tool `openvdb/tools/LevelSetMultiObject.h`

## 1. Goal

Port the multi-object geodesic active contour method implemented in wedge
(`wedge/src/core/geometry/MultiActiveContour3D.cpp`) to OpenVDB as a reusable,
upstream-quality tool.

The method represents N labeled objects with **one unsigned distance field φ
paired with one integer label field L**. For any label l the signed distance is

    ψ_l(x) = −φ(x)  if L(x) == l
    ψ_l(x) = +φ(x)  otherwise

The pair is evolved together so that, at any time, an accurate signed distance
for any label can be reconstructed near that label's boundary. By construction
the labeled regions partition space: no gaps, no overlaps.

### Success criteria

- The φ/L pair is stored in standard OpenVDB grid types (`FloatGrid`, `Int32Grid`).
- After every step, `extractSignedDistance(l)` returns a signed distance for
  label l that is second-order accurate in the interface layer (verified by a
  convergence test, slope ≥ 1.8).
- A single-object MOGAC run matches OpenVDB's existing single level set
  evolution on the same problem.
- Eight distinct objects in a vortex flow collide and pack against each other
  while the partition invariant holds at every step.
- A multi-label contouring tool produces, for every label, a closed 2-manifold
  triangle mesh whose topology (components, genus) matches the label's voxel
  region under a chosen digital connectivity pair, evaluated directly on the
  φ/L pair at a cost proportional to interface area.

### Non-goals (v1)

- Shared / crack-free meshing of interfaces *between* labels (one face used
  by two labels, triple-junction curves). Each label is meshed independently;
  adjacent labels' surfaces may differ by up to a voxel near junctions.
- Mesh adaptivity / decimation (as in `VolumeToMesh` adaptivity).
- Non-uniform voxel sizes / non-linear transforms.
- Higher-order temporal integration (TVD-RK). Forward Euler only.
- Topology preservation by default (it is opt-in per label, §4.7).
- NanoVDB / GPU implementation.

## 2. Reference algorithm (wedge MCLS)

Per step, for active voxels with φ ≤ 0.5:

1. Candidate labels = own label plus labels of the 6 face neighbors.
2. For each candidate l compute dψ_l/dt from curvature (clamped to ±10),
   upwind pressure and upwind advection, evaluated on ψ_l from the previous
   state. Label 0 (background) has speed 0. Labels without a force index
   expand at constant speed 0.999.
3. Time step from CFL: dt = maxStep · min(1, 0.999 / max|F|).
4. v_l = ψ_l + dt·F_l. New label = argmin v_l (ties → smaller label);
   φ = |(v₁ − v₂)/2| using the two smallest values, or |v₁| if only one.
5. Redistance outer layers (first order, 3 passes), plug orphan voxels, swap
   buffers, shrink/grow the narrow band (half-width 3.5).

This design keeps steps 1–4 and replaces step 5 with a second-order,
label-aware redistancing and OpenVDB-native band maintenance.

## 3. Architecture

All new code lives in `openvdb/tools/LevelSetMultiObject.h`, namespace
`openvdb::OPENVDB_VERSION_NAME::tools`. Internal helpers go in a nested
`lsmo_internal` namespace.

### 3.1 `MultiLevelSet<FloatTreeT = FloatTree, LabelTreeT = Int32Tree>`

Data holder for the pair.

- Members: `typename Grid<FloatTreeT>::Ptr phi`,
  `typename Grid<LabelTreeT>::Ptr labels`, `int halfWidth` (in voxels).
- φ background = `halfWidth · dx`; label background = 0. Both grids share one
  transform. `phi` grid class is `GRID_UNKNOWN` (it is unsigned, not a level
  set) with grid metadata `"multi_level_set_role" = "phi"` / `"labels"`.
- **Invariant:** identical active topology in both trees, φ ≥ 0 and finite at
  every active voxel.
- API:
  - `static Ptr fromLabelGrid(const Grid<LabelTreeT>& labels, int halfWidth = 3)`
    — label 0 voxels are background; initial φ is built by seeding interfaces
    at voxel face midpoints (φ = 0.5·dx) then redistancing (section 5).
  - `static Ptr fromSignedGrids(const std::map<int32_t, typename Grid<FloatTreeT>::ConstPtr>&, int halfWidth = 3)`
    — each input must be a `GRID_LEVEL_SET` on a common transform. Label at a
    voxel = the label with the most negative ψ (or 0 if all ψ > 0);
    φ is computed with the two-smallest rule, then redistanced.
  - `float signedDistance(const Coord& ijk, int32_t label) const` — raw ±φ
    (accurate only where l's boundary is the nearest boundary; see 5.3).
  - `typename Grid<FloatTreeT>::Ptr extractSignedDistance(int32_t label, int halfWidth = -1) const`
    — exact per-label reconstruction (section 5.3). Returns a
    `GRID_LEVEL_SET` grid.
  - `std::vector<int32_t> labelSet() const` — sorted distinct non-zero labels.
  - `bool validate(std::string* message = nullptr) const`.
  - `void dilate(int iterations)`, `void erode(int iterations)`,
    `void pruneInactive()` — always applied to both trees together.

### 3.2 `LabelAwareStencil<MultiLevelSetT>`

Read-only accessor that, for a center voxel and a label l, fills a 19-point
neighborhood of ψ_l values (6 face + 12 edge + center) from the *time-n*
buffers. It exposes the same accessors as `math::SevenPointStencil` /
`math::GradStencil` (`getValue<i,j,k>()`, `gradient`, `normSqGrad`,
`meanCurvature`) so `math::` upwind-gradient and curvature operators can be
reused unchanged.

### 3.3 Speed functor concept

```cpp
struct SpeedFunctor {
    // dψ_l/dt at voxel ijk for candidate label l. Positive grows ψ (shrinks l).
    float operator()(const Coord& ijk, int32_t label, const LabelAwareStencil<...>& s) const;
    bool  isFixed(int32_t label) const;   // speed forced to 0 (walls, background)
    bool  isActive(int32_t label) const;  // false → constant expansion speed
    float stableDt(float dx) const;       // extra stability bound (e.g. dx²/(6w)
                                          // for curvature); +inf if none
};

Units: ψ, φ and speeds are world space; dt is world time.
dt = min(maxDt, cfl·dx / max|F|, speed.stableDt(dx)). Curvature uses
κ = (k₁+k₂)/2, matching `math::CurvatureStencil::meanCurvatureNormGrad`
and wedge.
```

Label 0 is always fixed regardless of `isFixed`.

### 3.4 Built-in `MultiObjectForces<MultiLevelSetT>`

Reproduces wedge's force model:

- `setCurvatureWeight(float)`, `setMaxCurvatureForce(float = 10)`.
- `setPressure(const FloatGrid::ConstPtr&, float weight, float target = NaN)`
  — target rescaling as in wedge `rescale()` (shift by target, scale negative
  and positive parts independently to [−1, 1]), done once at set time into an
  internal copy.
- `setAdvection(const Vec3SGrid::ConstPtr&, float weight)` — sampled with
  `BoxSampler` in index space.
- `setInactiveLabels(std::set<int32_t>)`, `setFixedLabels(std::set<int32_t>)`.
- Speed = −advection·∇⁻ψ (upwind, per axis on sign of velocity component)
  + κ (clamped) + pressure·|∇ψ| (Godunov upwind on sign of pressure force),
  matching wedge's `pressureAndAdvectionMotion`.
- Deviation from wedge (ruling R26): the curvature term is evaluated at the
  interface rather than the voxel center, κ_Γ = κ_x / (1 − ψ·κ_x) with κ_x
  the voxel's mean curvature and ψ its ψ_l (falls back to κ_x when
  |ψ·κ_x| ≥ 0.5), then weighted and clamped as before.

### 3.5 `LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT = util::NullInterrupter>`

The evolver.

- `LevelSetMultiObject(MultiLevelSetT&, const SpeedT&, InterruptT* = nullptr)`
- `float step(float maxDt)` — one Euler step, returns dt used (world time).
- `size_t advect(double t0, double t1)` — steps until t1; returns step count.
  Checks interrupter each step; on interrupt returns with valid state.
- Setters: `setCFL(float = 0.999)`, `setGrainSize(int)`,
  `setRedistanceTolerance(float)`.
- Uses `tree::LeafManager` with one auxiliary buffer on each tree; reads from
  the main buffers (time n), writes to the aux buffers, then swaps both.

### 3.6 `MultiLevelSetRedistance<MultiLevelSetT, InterruptT>`

Standalone second-order label-aware redistancer (section 5), used by the
evolver, the factories, and `extractSignedDistance`.

### 3.7 `LevelSetMultiObjectMesh.h` — topology-constrained multi-label contouring

Separate header (section 6). `VolumeToMesh.h` is not modified.

### 3.8 Other files

- `openvdb/unittest/TestLevelSetMultiObject.cc`,
  `openvdb/unittest/TestLevelSetMultiObjectMesh.cc` + registration in
  `openvdb/unittest/CMakeLists.txt`.
- `openvdb/CMakeLists.txt`: add both headers to the installed tools list.
- `pendingchanges/multi_object_level_set.txt`.
- Doxygen: file-level docs with usage example and literature references.

## 4. Evolution step

Forward Euler. All reads from time-n main buffers, writes to aux buffers.

1. **Interface set.** Voxels with φ ≤ 0.5·dx. Candidates = own label ∪ distinct
   face-neighbor labels. Fixed labels and label 0 participate with speed 0.
   Inactive labels get speed +0.999 (constant expansion in wedge sign
   convention, i.e. ψ of the *other* labels grows).
2. **Speeds.** F_l = `speed(ijk, l, stencil)` per (voxel, candidate). Stored per
   leaf in a compact candidate list (≤ 7 entries per voxel).
3. **CFL.** dt = min(maxDt, cfl·dx / max|F|, speed.stableDt(dx)) via
   `tbb::parallel_reduce`.
4. **Two-smallest projection.** v_l = ψ_l + dt·F_l; label = argmin (ties →
   smaller label); φ = |(v₁ − v₂)/2|, or |v₁| with a single candidate.
   Non-interface voxels copy their time-n values.
5. **Orphan cleanup.** A voxel whose label does not appear in its 26-neighborhood
   takes the label of its smallest-φ neighbor and is flagged for redistancing.
6. **Band maintenance.**
   a. Swap aux → main for both trees.
   b. Dilate both trees one face layer where any voxel with φ < (halfWidth − 1)·dx
      touches the band edge. New voxels take the label of their active face
      neighbor with smallest φ; φ = halfWidth·dx.
   c. Redistance (section 5).
   d. Deactivate voxels with φ > halfWidth·dx in both trees; `pruneInactive`.
   e. Debug builds: `assert(validate())`.

### 4.7 Topology-preserving evolution (opt-in, per label)

Added at the user's request (2026-11-09).

- API: `LevelSetMultiObject::setTopologyPreservingLabels(std::set<int32_t>)`
  (default empty) and `setConnectivity(MultiObjectConnectivity)`. The enum
  `MultiObjectConnectivity { FG26_BG6, FG6_BG26, FG18_BG6, FG6_BG18 }` lives in
  `LevelSetMultiObject.h` (default `FG26_BG6`) and is shared with the mesher
  (§6.2 uses it instead of a nested enum).
- After the two-smallest projection proposes label changes, a change a → b at
  voxel x is accepted only if x is a *simple point* for every protected label
  in {a, b}: removing x from region a and adding x to region b must each leave
  the region's topology unchanged under the connectivity pair. Simplicity is
  decided by the Bertrand–Malandain topological numbers (T_fg = 1 and
  T_bg = 1 on the 3×3×3 neighborhood), which preserve both the region and its
  complement (no new components, tunnels or cavities).
- Proposed changes are applied in 27 colored passes (voxel coordinates mod 3),
  so no two changes applied in the same pass share a 26-neighborhood; each
  pass tests simplicity against the labels as updated by earlier passes. The
  result is independent of thread count.
- A rejected change keeps the voxel's label; its φ is set to 1e-3·dx (a
  barrier just inside its own region) and the voxel is frozen for this step's
  redistancing.
- Every label write in a step respects protection: projection (above),
  orphan cleanup (§4.5: a relabel involving a protected label is applied only
  if simple) and band activation (§4.6b: a newly activated voxel whose stored
  inactive label or proposed label is protected keeps its stored label).
- Labels that are not protected behave exactly as before.

## 5. Second-order label-aware redistancing

Restores φ = distance to the voxel's own label boundary across the band while
keeping the interface fixed.

### 5.1 Interface seeding (subcell fix)

Voxel p (label a) is an interface voxel if some face neighbor q has label
b ≠ a. Its value is the Russo–Smereka subcell fix (Russo & Smereka 2000,
second order as analyzed by du Chéné, Min & Gibou 2008), applied to
ψ_a over p and its face neighbors: ψ_a = −φ at voxels of label a, +φ
elsewhere. Terms that involve an inactive neighbor are omitted from Δ_i
(only band edges of finite domains have them). A seed with no active face
neighbor of the other label keeps its value, and seed values are clamped to
the φ background. Per axis i, with ψ(−1), ψ(0) = ψ_p, ψ(+1) the values along that axis,

    Δ_i = max(|ψ(+1) − ψ(−1)|/2, |ψ(+1) − ψ(0)|, |ψ(0) − ψ(−1)|, ε) / dx
    φ_p = |ψ_p| / sqrt(Σ_i Δ_i²)

with ε = 10⁻⁶·dx. (This replaces an axis-crossing formula,
φ_p = (Σ 1/d_i²)^(−1/2), that was only first order: it has no d_i on an
axis whose crossing lies more than one voxel away, and so overestimated
oblique interfaces by up to ≈ 0.35·dx.) For the per-label reconstruction
(5.3) the region "label a" is given by the isInside predicate.

All seeds are computed from a read-only snapshot of φ so results are order
independent. Seeds are frozen for 5.2. Correct at triple junctions because any
different-label neighbor marks a seed and has the opposite sign in ψ_a.

### 5.2 Outer layers

Solve |∇φ| = 1 on non-seed band voxels with seeds as Dirichlet data:

- Second-order one-sided upwind differences (Chopp 2001) where two same-side
  upwind neighbors with smaller φ exist; first-order Godunov update otherwise.
- Parallel Jacobi iterations over leaves with double buffers until
  max|Δφ| < tolerance (default 1e-4·dx) or ⌈halfWidth·√3⌉ + 2 iterations.
- Non-seed voxels only have same-label face neighbors, so no label logic is
  needed here.
- Implementation step 1 evaluates whether `FastSweeping.h` internals can be
  reused with frozen seeds and an all-positive field; if not, use the
  dedicated band Jacobi kernel described above.

### 5.3 Exact per-label reconstruction

In the paired field a voxel of label b stores the distance to *b's* boundary,
which near triple junctions may be a b|c interface rather than l. Raw ±φ
therefore underestimates ψ_l outside l there.
`extractSignedDistance(l, halfWidth)`:

1. Build a boolean "is l" mask over the band (dilated by halfWidth if needed).
2. Seeds are the band voxels with a face neighbor across the l | not-l
   boundary. They keep the pair's φ and are not reseeded with 5.1, because the
   subcell fix is not a fixed point of already reinitialized φ. An l-side seed's φ
   is the distance to l's boundary. A not-l seed q stores the distance to its
   own label's boundary, which can differ from ψ_l only next to a third
   label. Two bounds are proven, given exact l-side distances:
   φ_q ≤ ψ_l(q), because l's boundary seen from q is part of q's own label
   boundary; and ψ_l(q) ≤ |q − p| − φ_p for every l voxel p, because the ball
   of radius φ_p about p lies in l. Every not-l seed with a face neighbor
   (active or not) of a third label (neither l nor its own label) is rebuilt
   by a local Godunov extension of ψ_l: per axis, take the smallest known
   neighbor value a_i (−φ for l-side seeds, +φ for other not-l seeds and
   rebuilt seeds once solved), then solve Σ(ψ − a_i)² = dx². Jacobi sweeps
   over these seeds only continue until the values are stable. Each result
   is clamped to [φ_q, min_p(|q − p| − φ_p)] over the l voxels p in the 5³
   neighborhood (rulings R17, R19). ψ_l is not smooth at junction lines, so
   these values are only first-order accurate there. The rule was chosen by
   measurement over random z-parallel and tilted junctions against
   alternatives (no clamp, gradient-jump trigger, 3³ trigger, l-side-only
   Godunov, bound alone).
3. Solve 5.2 on |ψ_l| with those seeds frozen, apply sign from the mask.
4. Return as `GRID_LEVEL_SET` with background halfWidth·dx.

The evolver's speed stencils use raw ±φ, which is accurate at the face-neighbor
interface voxels where speeds are evaluated.

## 6. Topology-constrained multi-label contouring

### 6.1 Requirement

For every requested label l, output a triangle mesh M_l that is:

- a closed, orientable 2-manifold (every edge shared by exactly two triangles,
  every vertex link a single disk, consistent outward winding);
- topologically consistent with the digital region R_l = {voxels with L == l}
  under a user-chosen connectivity pair (foreground, background) ∈
  {(26, 6), (6, 26), (18, 6), (6, 18)}; default (26, 6). Concretely: M_l has one
  component per boundary of a connected component of R_l (outer boundary plus
  one per enclosed background cavity), and each component's genus equals that
  of the corresponding digital surface.

Interfaces between labels are *not* shared (non-goal, section 1).

### 6.2 API

```cpp
struct MultiObjectMeshSettings {
    enum class Connectivity { FG26_BG6, FG6_BG26, FG18_BG6, FG6_BG18 };
    Connectivity connectivity = Connectivity::FG26_BG6;
    bool junctionCorrection = true;   // section 6.4
    bool computeNormals = false;
    int grainSize = 1;                // TBB grain size in blocks; <= 0 serial (R36)
};

struct LabelMesh {
    std::vector<Vec3s> points;        // world space
    std::vector<Vec3I> triangles;     // CCW = outward
    std::vector<Vec3s> normals;       // optional
};

template<typename MultiLevelSetT, typename InterruptT = util::NullInterrupter>
std::map<int32_t, LabelMesh>
multiObjectToMesh(const MultiLevelSetT&,
                  const std::vector<int32_t>& labels = {},   // empty = all
                  const MultiObjectMeshSettings& = {},
                  InterruptT* = nullptr);
```

### 6.3 Algorithm

Cells are the 2×2×2 voxel-center cubes of the index grid.

1. **Interface cells, one pass (ruling R36).** Cells are enumerated in
   blocks of leaf-DIM^3 minimum corners: every leaf of the label tree plus its
   seven lower neighbor blocks (a cell whose minimum corner lies one voxel
   below a leaf has corners in it), block-parallel. Each block reads its
   labels plus a one-voxel upper halo once and skips cells whose 8 corner
   labels agree, so every mixed cell is processed exactly once, including
   cells whose minimum corner lies in a tile. Interfaces inside tiles (absent
   in valid pairs) are not seen.
2. **Per-label classification.** For each distinct non-zero label l among a
   cell's corners with a mixed in/out pattern, the 8-bit case index is
   `bit_i = (L(corner_i) == l)`. Classification uses **labels, not the sign of
   φ**, so it is exactly the digital region R_l and immune to φ noise.
3. **Topology-consistent case table.** For each connectivity pair a 256-case
   triangulation table is *generated at static-init time* from a rule, not
   hand-typed:
   - face ambiguities (two diagonal in-corners on a face) are joined when the
     foreground connectivity is 18 or 26 and separated when it is 6;
   - interior (body-diagonal) ambiguities are joined only for 26 foreground,
     separated for 6 and 18;
   - symmetrically, for a 26 background (FG6_BG26) body-diagonal OUT-corner
     pairs are joined by a tube through the cube (the background stays
     connected);
   - the generator emits edge-vertex triangle fans per resolved surface patch
     with consistent winding.
   This rule set is the one shown by Lachaud & Montanvert to give closed
   manifold surfaces whose topology matches the digital object for these
   connectivity pairs. A unit test exhaustively checks all 256 cases per pair
   (each case's patches form disks with boundary on the cube faces matching the
   neighboring cube's face pattern), so adjacent cells always agree.
4. **Edge vertices.** On a cell edge between corners p (in l) and q (not in l)
   the vertex sits at θ·dx from p with θ = φ_p / (φ_p + φ_q), i.e. the linear
   zero crossing of raw ψ_l. Vertices are keyed by (global edge id, label) and
   deduplicated with leaf-local hash maps, then stitched across leaves in a
   final merge so each label's mesh is watertight.
5. **Output.** Thread-local per-label triangle buffers are concatenated per
   label; optional normals from the gradient of the cell's local ψ_l.

Cost: cells are enumerated per block of DIM^3 minimum corners: each label
leaf plus its seven lower-neighbor blocks (a cell whose minimum corner is one
voxel below a leaf has corners in it). A block reads a 9^3 label window once
and exits early when the window is uniform; otherwise work is proportional to
the number of (cell, label) pairs, about 2 per interface cell. Labels absent
from a region cost nothing; no per-label grid is built.

### 6.4 Junction correction

At a voxel q of label b next to a third label c, φ_q is the distance to b's
nearest boundary, which may be the b|c interface rather than l. That biases θ
in cells containing three or more labels. With `junctionCorrection` (default
true), on every edge of which at least one incident cell has three or more
distinct labels (label 0 included), the outside endpoint q gets
ψ_l(q) = one local Godunov step on ψ_l (per axis the smaller of the two
face-neighbor values, −φ in l, +φ elsewhere), clamped to the proven bounds of
rulings R17/R19: [φ_q, max(φ_q, U)] with U = min over label-l voxels p in the
5×5×5 neighborhood of |q − p|·dx − φ_p. The l-side φ_p is kept. The value
depends only on (q, l), so all cells sharing the edge agree. This affects
geometry only; topology is fixed by steps 2–3 (ruling R35; the original "5.1
seeding" rule was measured to change nothing). It lowers the worst-case vertex
error over the junction set (0.653 → 0.272·dx) but may worsen individual
junctions slightly where φ_q was already exact.

## 7. Error handling

- `OPENVDB_THROW(ValueError, …)`: mismatched transforms; non-uniform voxel
  size; halfWidth < 2; a `fromSignedGrids` input that is not `GRID_LEVEL_SET`;
  negative or non-finite φ on input; built-in forces missing a grid they were
  configured to use.
- Interrupter checked per step and in long leaf loops; state stays valid
  because both aux buffers are swapped together or not at all.
- `validate()` checks topology equality, φ ≥ 0, finiteness. Asserted after each
  step in debug builds.

## 8. Testing (GoogleTest)

Analytic / unit tests:

1. **Round trip.** `fromSignedGrids` with two disjoint spheres; per-label
   `extractSignedDistance` vs analytic distance: max error < 0.05·dx in the
   interface layer, RMS < 0.1·dx over the band.
2. **Convergence order.** Sphere at three resolutions; fitted interface-layer
   error slope ≥ 1.8.
3. **Triple junction.** Three half-spaces meeting at 120° along a straight
   junction line, over a seeded set of lines (z-parallel with random angle
   and sub-voxel offset, and tilted). `extractSignedDistance` matches analytic
   distance per label: max interface error < 0.05·dx for interface voxels at
   least 2 voxels from the junction line, < 0.5·dx within 2 voxels (ψ_l is
   not smooth at the line, so the method is first order there), and
   RMS < 0.1·dx over the whole band. Raw ±φ is shown to underestimate by more
   than 0.2·dx somewhere outside each label (regression guard for 5.3).
4. **Partition invariant.** N steps of random-coefficient forces: the
   interiors {ψ_l < 0} of the per-label `extractSignedDistance` grids are
   pairwise disjoint (up to 1e-3·dx) and their union with the background
   region covers the band (no gaps).
5. **Motion sanity.** Constant pressure grows a sphere by ∫F dt (± 0.5·dx);
   mean-curvature flow with κ = (k₁+k₂)/2 and weight w follows
   r² = r₀² − 2·w·t (3D, world units, within tolerance); uniform
   advection translates a sphere.
6. **Fixed labels.** A fixed label's voxels are bit-identical after N steps.
7. **Band maintenance.** Topology equality after dilate/erode; band width never
   exceeds halfWidth + 1.

Required integration tests:

8. **Single-object equivalence.** One sphere evolved by MOGAC (one label vs
   background) and by OpenVDB's single level set path on the same problem,
   with the reference pinned to `FIRST_BIAS` spatial and `TVD_RK1` temporal
   schemes:
   - advection only: `LevelSetAdvection` with the same uniform/rotational
     velocity field;
   - curvature only: mean-curvature flow via `LevelSetFilter::meanCurvature`
     iterations matched to the same total time;
   - combined advection + curvature.
   Compare `extractSignedDistance(1)` against the reference grid: zero-set
   Hausdorff distance < 1·dx in every case. Both methods are compared with
   the exact solution rather than with each other at 1%. Advection only
   (ruling R25; both first-order schemes diffuse by different amounts):
   MOGAC's symmetric difference against the exact (rotated) solution is at
   most 1.1 × the reference's symmetric difference against it. Curvature only
   and combined (ruling R28): the mean zero-crossing radius about the exact
   center (unmoved, or rotated analytically for combined) is compared with
   r² = r₀² − 2t; MOGAC's |radius error| ≤ max(1.1 × the reference's
   |radius error|, 0.25·dx). For combined the bound also admits the
   reference's |radius error| + 0.1·dx (ruling R30): the curvature limit
   stableDt = dx²/(6w) caps every MOGAC step and so forces its advection to
   CFL ≈ 0.13, where first-order upwinding is most diffusive; operator
   splitting (advection at its own CFL, curvature substepped) is a known
   follow-up.
9. **Eight objects in a vortex** (parameters per rulings R27 and O18). Eight
   shapes with **eight distinct labels** (sphere, cube, tetrahedron,
   octahedron, dodecahedron, icosahedron, capsule, closed tube ring; via
   `LevelSetSphere`, `LevelSetPlatonic`, `LevelSetTubes`), each of
   circumradius 14·dx (the original sizes times 1.4), dx = 1. They sit in pairs on four rays (angles k·90°)
   in the plane z = −20, one inside and one outside the ring r = 38.5, each
   with its nearest surface point 0.75·dx from the ring. Velocity field:
   Rankine-style swirl about z (core R = 16, amplitude 0.1), radial inflow
   towards the ring u_r = −0.1·tanh(r − 38.5), updraft 0.1·exp(−r²/(2R²)); the velocity grid is
   dense over [−80, 80]³.
   (An exact flow never makes disjoint objects touch and an axisymmetric
   flow keeps the angular gaps of objects at one radius, so contact comes
   from pairs pressed together in the convergence zone until their gap is
   below a voxel.) Advection weight 1, curvature weight 0.003; advect in
   chunks of 2 until ≥ 4 contact pairs (capped at t = 70), then on to
   t = 60 with the contacts required to persist at every chunk (rulings R31
   and O18). The run is long, so the field is weak: every object keeps
   translating towards the ring at the far-field inflow speed while its face
   at the ring cannot move, so the flow flattens it by inflow × (time in
   contact), and the original strength (inflow and updraft 0.2, circumradius
   10) lost the capsule by t = 58; halving the inflow and updraft, starting
   the pairs closer and enlarging the shapes keeps every label alive (first contact at t = 36;
   smallest label 1203 voxels at t = 60; tightest volume-gate margin 3.4 points,
   label 8 upper bound). The run was first t = 84, then shortened to t = 60.
   - Every chunk: `validate()`, partition invariant, no NaN.
   - At end: ≥ 4 distinct label pairs sharing a face; every label still owns
     at least one voxel; with refLoss the volume loss of the same shape
     advected alone by `LevelSetAdvection` (FIRST_BIAS/TVD_RK1) in the same
     field over the same time, each label's voxel-count loss lies in
     [1.1·min(refLoss, 0) − 2%, 1.1·max(refLoss, 0) + 2%] (ruling R31).
   - Different labels never merge; contact means shared interfaces.

Contouring tests (`TestLevelSetMultiObjectMesh.cc`):

10. **Case tables.** For each connectivity pair, all 256 cases: every patch is
    a disk, windings are consistent, and the boundary on each cube face depends
    only on that face's 4 corners (so neighboring cells always agree).
11. **Manifold + closed.** On the round-trip spheres, the vortex end state
    (test 9), and random label volumes: every edge has exactly two incident
    triangles, every vertex link is one disk, and there are no degenerate
    triangles.
12. **Digital topology match.** For each label, compute the foreground
    components (chosen fg connectivity) and enclosed background cavities (bg
    connectivity), plus the Euler characteristic of the digital object from
    its cubical complex. Assert the mesh's component count equals
    components + cavities and χ(M_l) = 2·χ(R_l). Fixtures: solid torus
    (genus 1); two cubes touching only along an edge and only at a corner
    (joined under 26/18 vs 26 only, separated under 6); hollow sphere (cavity);
    random noise labels.
13. **Geometry.** Sphere: vertices within 0.1·dx of the analytic surface.
    Triple junction (ruling R35): with `junctionCorrection` the worst-case max
    vertex error over the 12-junction set of test 3 and the max vertex error
    of the original test 3 fixture both drop below those of the uncorrected
    run; per-junction numbers are printed.
15. **Topology: cube through three spheres** (`TopologyCubeThroughSpheres`).
    A cube (label 1, side 40·dx) translates along +x by advection (velocity
    (1,0,0), weight 1) a distance of side + sphere diameter + 8·dx, so every
    sphere passes completely through it. Three stationary spheres (labels 2–4,
    radius 4·dx, fixed labels) lie inside the cube's swept path at distinct
    (y, z) positions. Label 1 is topology-preserving (FG26_BG6).
    - Every chunk: the cube region has exactly 1 component (26-adjacency), 0
      enclosed cavities (6-adjacency) and region Euler characteristic 1; sphere
      voxels are bit-identical; `validate()` and the partition invariant hold.
    - End: cube centroid moved by the commanded distance ± 2·dx.
    - Control: the same run without protection must change the cube's
      topology at some chunk (cavities > 0 or χ ≠ 1), proving the test
      discriminates.

14. **Efficiency (benchmark, non-gating).** Eight-label vortex state: time
    `multiObjectToMesh` against per-label `extractSignedDistance` +
    `volumeToMesh`. Report the speedup; the gating assertion is only
    "not slower than per-label".

## 9. Implementation order (for the plan)

1. `MultiLevelSet` holder + `validate` + factories (seeding only).
2. Redistancer (5.1, 5.2) + tests 1–2.
3. `extractSignedDistance` (5.3) + test 3.
4. `LabelAwareStencil` + `MultiObjectForces`.
5. Evolver step 1–6 + tests 4–7.
6. Integration tests 8–9.
7. Contouring: case-table generator + test 10, then `multiObjectToMesh` +
   tests 11–14.
8. Docs, CMake install list, pendingchanges.

## 10. References

- B. C. Lucas, M. Kazhdan, R. H. Taylor, "Multi-Object Geodesic Active
  Contours (MOGAC): How to Represent and Manipulate Multiple Level Sets with
  only a Label Mask and Distance Field," Medical Image Computing and Computer
  Assisted Intervention (MICCAI), 2012. This feature implements that paper.
- G. Russo, P. Smereka, "A remark on computing distance functions," JCP 2000.
- D. Chopp, "Some improvements of the fast marching method," SIAM J. Sci.
  Comput. 2001.
- J.-O. Lachaud, A. Montanvert, "Continuous analogs of digital boundaries: a
  topological approach to iso-surfaces," Graphical Models 2000.
