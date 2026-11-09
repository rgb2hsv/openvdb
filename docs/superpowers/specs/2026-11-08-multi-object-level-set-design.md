# Multi-Object Level Set (MOGAC) for OpenVDB — Design

Date: 2026-11-08
Status: Draft for review
Target: upstream OpenVDB, header-only tool `openvdb/tools/LevelSetMultiObject.h`

## 1. Goal

Port the multi-object geodesic active contour method implemented in wedge
(`wedge/src/core/geometry/MultiActiveContour3D.cpp`) to OpenVDB as a reusable,
upstream-quality tool.

The method represents N labelled objects with **one unsigned distance field φ
paired with one integer label field L**. For any label l the signed distance is

    ψ_l(x) = −φ(x)  if L(x) == l
    ψ_l(x) = +φ(x)  otherwise

The pair is evolved together so that, at any time, an accurate signed distance
for any label can be reconstructed near that label's boundary. By construction
the labelled regions partition space: no gaps, no overlaps.

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
- Topology preservation.
- NanoVDB / GPU implementation.

## 2. Reference algorithm (wedge MCLS)

Per step, for active voxels with φ ≤ 0.5:

1. Candidate labels = own label plus labels of the 6 face neighbours.
2. For each candidate l compute dψ_l/dt from curvature (clamped to ±10),
   upwind pressure and upwind advection, evaluated on ψ_l from the previous
   state. Label 0 (background) has speed 0. Labels without a force index
   expand at constant speed 0.999.
3. Time step from CFL: dt = maxStep · min(1, 0.999 / max|F|), unless speeds are
   clamped to ±1.
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

Read-only accessor that, for a centre voxel and a label l, fills a 19-point
neighbourhood of ψ_l values (6 face + 12 edge + centre) from the *time-n*
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
    float maxSpeedHint() const;           // optional; <0 if unknown
};
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

### 3.5 `LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT = util::NullInterrupter>`

The evolver.

- `LevelSetMultiObject(MultiLevelSetT&, const SpeedT&, InterruptT* = nullptr)`
- `float step(float maxDt)` — one Euler step, returns dt used (index-space
  time units scaled by dx).
- `size_t advect(double t0, double t1)` — steps until t1; returns step count.
  Checks interrupter each step; on interrupt returns with valid state.
- Setters: `setCFL(float = 0.999)`, `setClampSpeed(bool)`, `setGrainSize(int)`,
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
   face-neighbour labels. Fixed labels and label 0 participate with speed 0.
   Inactive labels get speed +0.999 (constant expansion in wedge sign
   convention, i.e. ψ of the *other* labels grows).
2. **Speeds.** F_l = `speed(ijk, l, stencil)` per (voxel, candidate). Stored per
   leaf in a compact candidate list (≤ 7 entries per voxel).
3. **CFL.** dt = min(maxDt, cfl·dx / max|F|) via `tbb::parallel_reduce`.
   With clampSpeed, F is clamped to [−1, 1] and dt = maxDt.
4. **Two-smallest projection.** v_l = ψ_l + dt·F_l; label = argmin (ties →
   smaller label); φ = |(v₁ − v₂)/2|, or |v₁| with a single candidate.
   Non-interface voxels copy their time-n values.
5. **Orphan cleanup.** A voxel whose label does not appear in its 26-neighbourhood
   takes the label of its smallest-φ neighbour and is flagged for redistancing.
6. **Band maintenance.**
   a. Swap aux → main for both trees.
   b. Dilate both trees one face layer where any voxel with φ < (halfWidth − 1)·dx
      touches the band edge. New voxels take the label of their active face
      neighbour with smallest φ; φ = halfWidth·dx.
   c. Redistance (section 5).
   d. Deactivate voxels with φ > halfWidth·dx in both trees; `pruneInactive`.
   e. Debug builds: `assert(validate())`.

## 5. Second-order label-aware redistancing

Restores φ = distance to the voxel's own label boundary across the band while
keeping the interface fixed.

### 5.1 Interface seeding (subcell fix)

Voxel p (label a) is an interface voxel if some face neighbour q has label
b ≠ a. Along that axis the zero crossing of ψ_a lies at θ·dx from p with
θ = φ_p / (φ_p + φ_q) (guard: if φ_p + φ_q < ε, θ = 0.5). For each axis take the
nearest crossing in ±, giving d_x, d_y, d_z (∞ if none); set

    φ_p = (Σ_i 1/d_i²)^(−1/2)

All seeds are computed from a read-only snapshot of φ so results are order
independent. Seeds are frozen for 5.2. Correct at triple junctions because any
different-label neighbour defines a crossing.

### 5.2 Outer layers

Solve |∇φ| = 1 on non-seed band voxels with seeds as Dirichlet data:

- Second-order one-sided upwind differences (Chopp 2001) where two same-side
  upwind neighbours with smaller φ exist; first-order Godunov update otherwise.
- Parallel Jacobi iterations over leaves with double buffers until
  max|Δφ| < tolerance (default 1e-4·dx) or ⌈halfWidth·√3⌉ + 2 iterations.
- Non-seed voxels only have same-label face neighbours, so no label logic is
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
2. Seed with 5.1 using only l | not-l crossings.
3. Solve 5.2 on |ψ_l|, apply sign from the mask.
4. Return as `GRID_LEVEL_SET` with background halfWidth·dx.

The evolver's speed stencils use raw ±φ, which is accurate at the face-neighbour
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

Cells are the 2×2×2 voxel-centre cubes of the index grid.

1. **Interface cells, one pass.** Leaf-parallel over band voxels that have a
   face neighbour of a different label (the same set as redistancing seeds).
   Each such voxel contributes the up-to-8 cells it is a corner of. Cells are
   deduplicated per leaf by owning-corner rule (the cell belongs to the leaf
   containing its min corner). For each cell, read the 8 corner labels once.
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
   - the generator emits edge-vertex triangle fans per resolved surface patch
     with consistent winding.
   This rule set is the one shown by Lachaud & Montanvert to give closed
   manifold surfaces whose topology matches the digital object for these
   connectivity pairs. A unit test exhaustively checks all 256 cases per pair
   (each case's patches form disks with boundary on the cube faces matching the
   neighbouring cube's face pattern), so adjacent cells always agree.
4. **Edge vertices.** On a cell edge between corners p (in l) and q (not in l)
   the vertex sits at θ·dx from p with θ = φ_p / (φ_p + φ_q), i.e. the linear
   zero crossing of raw ψ_l. Vertices are keyed by (global edge id, label) and
   deduplicated with leaf-local hash maps, then stitched across leaves in a
   final merge so each label's mesh is watertight.
5. **Output.** Thread-local per-label triangle buffers are concatenated per
   label; optional normals from the gradient of the cell's local ψ_l.

Cost: one read of each interface voxel's 8-cell neighbourhood, plus work
proportional to the number of (cell, label) pairs, which is about 2 per
interface cell. Labels absent from a region cost nothing; no per-label grid
is built.

### 6.4 Junction correction

At a voxel q of label b next to a third label c, φ_q is the distance to b's
nearest boundary, which may be the b|c interface rather than l. That biases θ
in cells containing three or more labels. With `junctionCorrection`, for such
cells only, recompute local ψ_l at the 8 corners using the 5.1 seeding
restricted to l | not-l crossings (a 4×4×4 stencil) before placing vertices.
This affects geometry only; topology is fixed by step 2–3.

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
3. **Triple junction.** Three half-spaces meeting at 120°. `extractSignedDistance`
   matches analytic distance per label; raw ±φ is shown to underestimate
   (regression guard for 5.3).
4. **Partition invariant.** N steps of random-coefficient forces: the
   interiors {ψ_l < 0} of the per-label `extractSignedDistance` grids are
   pairwise disjoint (up to 1e-3·dx) and their union with the background
   region covers the band (no gaps).
5. **Motion sanity.** Constant pressure grows a sphere by ∫F dt (± 0.5·dx);
   mean-curvature flow follows r² = r₀² − 4t (3D, within tolerance); uniform
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
   Compare `extractSignedDistance(1)` against the reference grid: symmetric
   difference volume < 1% of object volume and zero-set Hausdorff distance
   < 1·dx.
9. **Eight objects in a tornado vortex.** Eight shapes with **eight distinct
   labels** (sphere, cube, tetrahedron, octahedron, dodecahedron, icosahedron,
   capsule, torus-like tube; via `LevelSetSphere`, `LevelSetPlatonic`,
   `LevelSetTubes`), spaced apart in a ring around the z axis. Velocity field:
   Rankine-style swirl about z (solid-body core, 1/r outer decay), radial
   inflow near the bottom, updraft near the core. Evolve with advection plus a
   small curvature weight.
   - Every N steps: `validate()`, partition invariant, no NaN.
   - At end: objects have come into contact (count of distinct label pairs
     sharing a face > 0, expected ≥ 4); per-label volume drift within a stated
     tolerance (velocity is near divergence-free); every label still owns
     at least one voxel (no label vanishes).
   - Different labels never merge; contact means shared interfaces.

Contouring tests (`TestLevelSetMultiObjectMesh.cc`):

10. **Case tables.** For each connectivity pair, all 256 cases: every patch is
    a disk, windings are consistent, and the boundary on each cube face depends
    only on that face's 4 corners (so neighbouring cells always agree).
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
    Triple junction (test 3 fixture): with `junctionCorrection` the max vertex
    error drops below that of the uncorrected run.
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

- B. C. Lucas, M. Kazhdan, R. H. Taylor: the MOGAC (multi-object geodesic
  active contours) and MUSCLE (multi-object spring level sets) papers.
  Exact venue/year to be filled in by the author before upstreaming.
- G. Russo, P. Smereka, "A remark on computing distance functions," JCP 2000.
- D. Chopp, "Some improvements of the fast marching method," SIAM J. Sci.
  Comput. 2001.
- J.-O. Lachaud, A. Montanvert, "Continuous analogs of digital boundaries: a
  topological approach to iso-surfaces," Graphical Models 2000.
