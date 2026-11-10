# MOGAC optimizations — Design addendum

Date: 2026-11-09
Status: Draft for review
Branch: `feature/mogac-optimizations` (from `feature/multi-object-level-set` at 24b07a5f)
Base design: `docs/superpowers/specs/2026-11-08-multi-object-level-set-design.md`
Paper: B. C. Lucas, M. Kazhdan, R. H. Taylor, "Multi-Object Geodesic Active
Contours (MOGAC)", MICCAI 2012.

## 1. Scope (from the user)

Implement every speed and memory recommendation and every accuracy
recommendation; closest-point junction reconstruction is optional and ON by
default (user decision, 2026-11-09); labels default to 16 bit. API restructuring (renames, header split)
is NOT in scope.

| Id | Area | Item |
|----|------|------|
| S1 | speed + accuracy | Strang operator splitting of hyperbolic (advection, pressure) and parabolic (curvature) terms |
| S2 | speed | Incremental (dirty-region) redistancing with warm start |
| S3 | speed | Full-voxel interface motion per step (opt-in until proven) |
| S4 | speed + memory | LeafManager auxiliary buffers instead of per-step deep tree copies |
| S5 | speed | Direct field reads when transforms match; per-voxel field cache reused across candidates (exists) and across RK stages |
| A1 | accuracy | HJWENO5 upwinding with a label-aware ±3 stencil, and TVD-RK2 with the projection at each stage |
| A2 | accuracy | Speed extension: field-sampled speeds evaluated at the closest interface point |
| A3 | accuracy | Closest-point reconstruction at triple-junction seeds (extraction and mesher), on by default, can be turned off |
| A4 | accuracy | Optional per-label volume constraint (Lagrange-multiplier pressure) |
| M1 | memory | 16-bit labels by default |
| M2 | memory | Pooled per-leaf scratch (candidate arrays, freeze/dirty masks) reused across steps |
| M3 | memory | = S4 (no per-step copy of both trees) |

## 2. Design

### S1 Operator splitting

- The speed functor concept gains two optional members:
  `hyperbolic(ijk, label, stencil)` (first-order-in-space terms: advection,
  pressure, constant/inactive speeds) and `parabolic(ijk, label, stencil)`
  (curvature). A functor that only provides `operator()` is treated as fully
  hyperbolic (current behavior). `MultiObjectForces` provides both.
- `step(maxDt)` with a split functor performs Strang splitting:
  parabolic(dt/2) → hyperbolic(dt) → parabolic(dt/2). The hyperbolic part uses
  dt = min(maxDt, motion cap / max|F_h|); each parabolic half is subcycled with
  n = ⌈(dt/2) / stableDt⌉ substeps. Every sub-update is a full MOGAC update
  (projection, freeze, incremental redistance S2, band maintenance, topology
  protection), so the partition and topology guarantees are unchanged.
- `setSplitting(bool)` (default true when the functor is split).
- Acceptance: `SingleObjectCombined` returns to t = 15 with its current gate
  (radius error ≤ max(1.1·ref, ref + 0.1·dx, 0.25·dx)); measured MOGAC error at
  t = 10 and t = 15 must not exceed the reference by more than 0.1·dx.

### S2 Incremental redistancing

- Dirty mask = dilate(projected voxels ∪ voxels whose label changed ∪ newly
  activated voxels, halfWidth) ∩ band. Only dirty, non-frozen voxels are
  reset and solved; all other band voxels keep their values and act as
  Dirichlet data at the dirty-region boundary.
- Warm start: dirty voxels restart from min(previous value, background)
  rather than FLT_MAX where the previous value is still an upper bound
  (voxels whose nearest interface did not move toward them); others restart
  from FLT_MAX as now.
- Must be bit-identical to the full solve within tolerance 1e-4·dx on every
  existing evolution test (add a debug option `setFullRedistance(true)` used
  by a comparison test).

### S3 Full-voxel motion (opt-in)

- `setMaxInterfaceMotion(double voxels)`, default 0.5; 1.0 allowed.
- At 1.0 the projected set grows by one ring (φ ≤ 1.5·dx or within one voxel of
  a different label). Ring-2 voxels receive candidate labels from their
  26-neighborhood with ψ_l estimated by the closest-point estimate of A3
  (not +φ_q, which caused the earlier label race, nor the Lipschitz bound,
  which slowed motion).
- Gate for making 1.0 the default: ContactNoCascade (probe geometry: no
  cascade), PressureGrowth, UniformAdvection, CurvatureFlow, Vortex and the
  topology tests all pass at 1.0 with unchanged gates. If not, 0.5 stays the
  default and the result is reported.

### S4 / M3 Auxiliary buffers

- Replace the per-step deep copies of the φ and label trees with one
  `tree::LeafManager` auxiliary buffer per tree (allocated once per step only
  for leaves in the band, swapped together). Topology changes (band
  maintenance) happen after the swap, as now.
- Peak memory per step drops from ~2× the pair to ~1× + aux buffers of active
  leaves only.

### S5 Field reads

- If a velocity/pressure grid's transform equals the pair's transform,
  read voxel values directly (accessor `getValue`) instead of `BoxSampler`.
- With A2, sample once per voxel at the closest point and reuse across
  candidate labels and RK stages.

### A1 Higher-order upwinding

- `LabelAwareStencil` reads ψ_l on a ±3 axis stencil (19 + 12 points) when the
  spatial scheme is HJWENO5; values outside the band read ±background by label.
- `setSpatialScheme(math::BiasedGradientScheme)` — FIRST_BIAS or HJWENO5_BIAS
  (default HJWENO5_BIAS); `setTemporalScheme(math::TemporalIntegrationScheme)` —
  TVD_RK1 or TVD_RK2 (default TVD_RK2). For RK2 the projection (two-smallest,
  passive label 0, fixed walls, topology protection) is applied at each stage;
  stage 2 averages the stage values before the final projection.
- halfWidth ≥ 3 required for HJWENO5 (throw ValueError otherwise).
- Existing equivalence tests pin FIRST_BIAS/TVD_RK1 explicitly (they compare
  against OpenVDB's first-order reference); new tests show HJWENO5/RK2 reduces
  advection error and vortex volume loss.

### A2 Speed extension

- For field-sampled terms (advection velocity, pressure), sample at the
  closest interface point x_Γ = x − ψ·∇ψ/|∇ψ|² (world space) instead of the
  voxel center. Curvature keeps the interface correction already in place.
- Gate: CurvatureFlow / SingleObjectCurvature bias does not get worse; the
  distance property after evolution (ExtractAccuracyAfterEvolution) improves
  or stays within its gate.

### A3 Closest-point junction reconstruction (on by default)

- `ExtractSettings { int halfWidth = -1; int grainSize = 1; bool
  closestPointJunctions = true; }` for `extractSignedDistance`, and
  `MultiObjectMeshSettings::closestPointJunctions = true` for the mesher.
- When on, a not-l seed next to a third label gets ψ_l from the closest point
  on a local piecewise-planar reconstruction of ∂l built from l-side seeds
  (plane through p − φ_p·n_p with normal n_p from one-sided ψ_l differences
  inside l), clamped to the proven bounds [φ_q, min_p(|q−p| − φ_p)].
- Gate on the 12-junction set: worst interface error within 2 voxels of a
  junction line improves over the current rule; far-field unchanged.

### A4 Per-label volume constraint (optional)

- `setVolumeConstraint(int32_t label, double targetVolume)` and
  `preserveInitialVolume(int32_t label)`; off by default.
- Each step adds a uniform normal speed λ_l to label l's hyperbolic speed with
  λ_l = (V_l − V_target) / (A_l · dt) · relaxation (default 0.5), where V_l is
  the label's volume (voxel count·dx³ plus interface sub-voxel correction from
  φ) and A_l its interface area estimate.
- Gate: vortex with all labels constrained keeps each label's volume within
  2% of its initial value; unconstrained labels unaffected.

### M1 16-bit labels by default

- `using Int16LabelTree = tree::Tree4<int16_t, 5, 4, 3>::Type;` in
  `LevelSetMultiObject.h`; `MultiLevelSet<FloatTreeT = FloatTree,
  LabelTreeT = Int16LabelTree>`.
- `registerMultiLevelSetGridTypes()` registers `Grid<Int16LabelTree>` for file
  I/O (OpenVDB does not register int16 grids by default); documented.
- Factories accepting label grids of a wider type check the range and throw
  ValueError for labels outside [−32768, 32767].
- Tests that exercise large label ids (e.g. 1<<30) instantiate the Int32
  variant explicitly; all other tests use the default.

### M2 Pooled scratch

- The evolver owns per-leaf scratch (candidate arrays, freeze/dirty masks,
  aux buffers) sized to the band and reused across steps; reallocated only
  when the leaf count grows.

## 3. Testing additions

- SplittingCombinedT15 (= SingleObjectCombined at t = 15), splitting on.
- IncrementalRedistanceMatchesFull (all evolution scenarios, ≤ 1e-4·dx).
- FullVoxelMotion suite (S3 gates above).
- AuxBuffersMemory: peak bytes during step ≤ 1.3× pair memory (via
  tree.memUsage before/after; documented approximation).
- FieldDirectReadMatchesSampler (aligned transforms: identical results).
- Weno5ReducesError (advection and vortex volume loss vs FIRST_BIAS).
- SpeedExtensionCurvature / ExtractAccuracyAfterEvolution (gates above).
- ClosestPointJunctions (12-junction set).
- VolumeConstraintVortex (≤ 2%).
- Int16LabelsDefault, Int16RangeThrows, Int16GridIO (register + write + read).
- A benchmark printing per-step time before/after (informational).

## 4. Order and risk

S4/M3/M2 first (pure refactor, bit-identical), then S5, S2, M1, S1, A1, A2,
A3, A4, S3 last (highest risk; may remain opt-in).
