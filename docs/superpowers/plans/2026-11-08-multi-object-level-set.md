# Multi-Object Level Set (MOGAC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an upstream-quality, header-only OpenVDB tool that evolves N labelled objects as one unsigned distance `FloatGrid` paired with one `Int32Grid` of labels, reconstructs a second-order signed distance per label at any time, and contours every label to a topology-correct manifold mesh.

**Architecture:** `LevelSetMultiObject.h` holds the φ/L pair (`MultiLevelSet`), a label-aware second-order redistancer, a label-aware stencil, a speed-functor concept with wedge-equivalent built-in forces, and a forward-Euler evolver using wedge's two-smallest projection. `LevelSetMultiObjectMesh.h` holds a rule-generated, connectivity-consistent 256-case table and a leaf-parallel mesher over interface cells.

**Tech Stack:** C++17, OpenVDB (`tree::LeafManager`, `tools::Morphology`, `math::` stencils), TBB, GoogleTest, CMake.

**Spec:** `docs/superpowers/specs/2026-11-08-multi-object-level-set-design.md` — read it alongside this plan; section numbers below (§) refer to it.

## Global Constraints

- Header-only; namespace `openvdb::OPENVDB_VERSION_NAME::tools`; internals in `tools::lsmo_internal`.
- Every file starts with `// Copyright Contributors to the OpenVDB Project` / `// SPDX-License-Identifier: Apache-2.0`.
- φ, ψ, speeds in world units; dt in world time; uniform voxel size `dx` only.
- φ background = `halfWidth·dx`; label background = 0; label 0 is always fixed.
- Default `halfWidth = 3`, must be ≥ 2; default `cfl = 0.999`; default redistance tolerance `1e-4·dx`; default max curvature force 10.
- Curvature convention κ = (k₁+k₂)/2 (as `math::CurvatureStencil::meanCurvatureNormGrad`).
- Ties in argmin go to the smaller label.
- φ and L trees always have identical active topology; every topology op touches both.
- Multithreading via TBB with a settable grain size; grain size ≤ 0 runs serially.
- Errors via `OPENVDB_THROW(ValueError, …)`; interrupts via `InterruptT` (default `util::NullInterrupter`).
- `VolumeToMesh.h`, `LevelSetTracker.h` and other existing headers are not modified.

## Review Focus

1. **Voxel size ≠ 1** — every distance, speed and tolerance scales with dx; expect identical index-space results at dx = 0.25. Pinned in Task 3 (`RoundTripNonUnitVoxel`).
2. **Absent or empty labels** — `extractSignedDistance` of a label not present returns an all-background grid (all ψ > 0, no active voxels); `step()` on a pair with only background returns `maxDt` and changes nothing; mesher returns an empty map. Pinned in Tasks 4, 6, 10.
3. **Negative and large label IDs** (e.g. −7, 2³⁰) — treated as ordinary labels everywhere; only 0 is special. Pinned in Tasks 1 and 10.
4. **Interrupt mid-advect** — `advect` returns the completed step count and `validate()` is true afterwards. Pinned in Task 6.
5. **Sub-voxel features** (an object one voxel thick shrinking under curvature) — the label may disappear but no NaN, no orphan voxel, `validate()` holds. Pinned in Task 6.

---

## Build & run (used by every task)

Configure once:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOPENVDB_BUILD_UNITTESTS=ON \
  -DOPENVDB_TESTS="LevelSetMultiObject;LevelSetMultiObjectMesh"
```

Build + run a filter:

```bash
cmake --build build --target vdb_test -j && build/openvdb/openvdb/unittest/vdb_test --gtest_filter='<Filter>'
```

Below, "Run `<Filter>`" means the second command with that filter.

---

### Task 1: `MultiLevelSet` holder and registration

**Files:**
- Create: `openvdb/openvdb/tools/LevelSetMultiObject.h`
- Create: `openvdb/openvdb/unittest/TestLevelSetMultiObject.cc`
- Modify: `openvdb/openvdb/unittest/CMakeLists.txt` (add `TestLevelSetMultiObject.cc` and `TestLevelSetMultiObjectMesh.cc` alphabetically after `TestLevelSetFilter.cc`)
- Modify: `openvdb/openvdb/CMakeLists.txt` (add `tools/LevelSetMultiObject.h`, `tools/LevelSetMultiObjectMesh.h` to the tools header list near `tools/LevelSetTracker.h`)

**Interfaces:**
- Produces:
  ```cpp
  template<typename FloatTreeT = FloatTree, typename LabelTreeT = Int32Tree>
  class MultiLevelSet {
  public:
      using Ptr = SharedPtr<MultiLevelSet>;
      using FloatGridT = Grid<FloatTreeT>; using LabelGridT = Grid<LabelTreeT>;
      MultiLevelSet(typename FloatGridT::Ptr phi, typename LabelGridT::Ptr labels, int halfWidth); // throws on mismatch
      typename FloatGridT::Ptr phi(); typename LabelGridT::Ptr labels(); // + const overloads
      int halfWidth() const; double voxelSize() const;
      float signedDistance(const Coord& ijk, int32_t label) const;   // raw ±φ
      std::vector<int32_t> labelSet() const;                           // sorted, non-zero
      bool validate(std::string* message = nullptr) const;
      void dilate(int iterations); void erode(int iterations); void pruneInactive();
  };
  ```
  Metadata: φ grid `"multi_level_set_role"="phi"`, label grid `"labels"`; φ grid class `GRID_UNKNOWN`.

- [ ] **Step 1: Write failing tests** in `TestLevelSetMultiObject.cc` (fixture `TestLevelSetMultiObject` with `openvdb::initialize/uninitialize`, as in `TestLevelSetTubes.cc`). Helper `makePair(dx)` builds a 16³ dense block: labels 1 for x<8, 2 for x≥8, φ = |x − 7.5|·dx clamped to 3·dx.
  - `HolderValidate`: `validate()` true; grids carry `"multi_level_set_role"` metadata `"phi"`/`"labels"`; set one φ value to −1 → false with a non-empty message; activate a voxel only in L → false.
  - `HolderThrows`: mismatched transforms → `ValueError`; `halfWidth = 1` → `ValueError`; non-uniform scale transform → `ValueError`.
  - `HolderSignedDistance`: at (3,0,0) label 1 → `−φ`, label 2 → `+φ`.
  - `HolderLabelSet`: `{1,2}`; after relabelling some voxels to −7 and 1<<30 → `{−7, 1, 2, 1<<30}` (Review Focus 3).
  - `HolderDilateErode`: `dilate(2)` then `erode(2)`: active-voxel counts equal between trees after each, `validate()` true.
- [ ] **Step 2: Run `TestLevelSetMultiObject.Holder*`** — expected: compile failure (header missing).
- [ ] **Step 3: Implement `MultiLevelSet`.** `dilate/erode` call `tools::dilateActiveValues`/`erodeActiveValues` with `NN_FACE, IGNORE_TILES` on both trees; new φ voxels get background, new label voxels copy the label of the active face neighbour with smallest φ (do this in a post-pass over newly active voxels). `validate` compares `tree.hasSameTopology()` and scans φ for `< 0` or non-finite.
- [ ] **Step 4: Run `TestLevelSetMultiObject.Holder*`** — expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add MultiLevelSet holder for paired phi/label grids"`

---

### Task 2: Label-aware redistancer (§5.1, §5.2)

**Files:**
- Modify: `openvdb/openvdb/tools/LevelSetMultiObject.h`
- Test: `openvdb/openvdb/unittest/TestLevelSetMultiObject.cc`

**Interfaces:**
- Consumes: `MultiLevelSet` (Task 1).
- Produces:
  ```cpp
  template<typename MultiLevelSetT, typename InterruptT = util::NullInterrupter>
  class MultiLevelSetRedistance {
  public:
      explicit MultiLevelSetRedistance(MultiLevelSetT&, InterruptT* = nullptr);
      void setTolerance(float tolWorld);   // default 1e-4*dx
      void setGrainSize(int);
      int  redistance();                   // returns Jacobi iterations used
      // Seeding only, exposed for extractSignedDistance (Task 4):
      template<typename IsInsideOp>        // bool(const Coord&) -> voxel in target region
      static void seedInterface(const typename MultiLevelSetT::FloatGridT::TreeType& phiSnapshot,
                                IsInsideOp, typename MultiLevelSetT::FloatGridT::TreeType& out,
                                typename MultiLevelSetT::FloatGridT::TreeType::template ValueConverter<bool>::Type& seedMask);
  };
  ```

- [ ] **Step 1: Write failing tests.**
  - `RedistancePlanar`: a pair with interface at the plane x = 7.3 (labels 1 | 2), φ initialised to garbage `0.5·dx` everywhere in a 4-voxel band. After `redistance()`: every active voxel's φ equals `|x − 7.3|·dx` within `1e-3·dx`; iterations ≤ ⌈3·√3⌉ + 2 = 8.
  - `RedistanceKeepsCrossing`: same setup with φ already exact; after `redistance()`, the linear zero crossing between x=7 and x=8 is still 7.3 within `1e-5`.
  - `RedistanceOrderIndependent`: grain size 0 vs 1 gives bit-identical φ.
- [ ] **Step 2: Run `TestLevelSetMultiObject.Redistance*`** — expected: FAIL (undefined).
- [ ] **Step 3: Implement.**
  - Seeding per §5.1 from a deep-copied φ snapshot; θ guard ε = 1e-6·dx; seeds marked in a `BoolTree` mask and frozen.
  - Outer layers: parallel Jacobi over `LeafManager` with one aux buffer. Per voxel, per axis take the smaller-φ neighbour `a_i` (and the next one out `a_i'` on that side). Use the second-order one-sided value `(4a_i − a_i')/3` with coefficient `9/4` when `a_i' ≤ a_i`, else first order. Solve the Godunov quadratic `Σ c_i (φ − ā_i)² = dx²` over the sorted upwind set, dropping terms with `φ ≤ ā_i`. Keep `min(old, new)`. Stop at max|Δφ| < tolerance or ⌈halfWidth·√3⌉ + 2 iterations.
  - Before writing the kernel, spend ≤ 30 min checking whether `FastSweeping.h` internals take frozen seeds on an all-positive field (§5.2). Record the outcome in a code comment either way.
- [ ] **Step 4: Run `TestLevelSetMultiObject.Redistance*`** — expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add second-order label-aware redistancing"`

---

### Task 3: Factories + accuracy tests (spec tests 1–2)

**Files:** Modify `LevelSetMultiObject.h`; test in `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Consumes: `MultiLevelSetRedistance::redistance()` (Task 2).
- Produces (static members of `MultiLevelSet`):
  ```cpp
  static Ptr fromLabelGrid(const LabelGridT& labels, int halfWidth = 3);
  static Ptr fromSignedGrids(const std::map<int32_t, typename FloatGridT::ConstPtr>&, int halfWidth = 3);
  ```
  Test helper (file-local in tests, reused by later tasks): `double maxInterfaceError(const FloatGrid& sdf, std::function<double(Vec3d)> analytic)` and `double rmsBandError(...)`, both returning errors in units of dx. "Interface layer" = active voxels with a face neighbour of opposite sign.

- [ ] **Step 1: Write failing tests.**
  - `RoundTripTwoSpheres`: spheres r = 10 at (−15,0,0) and (15,0,0), dx = 1, via `createLevelSetSphere`, labels 1 and 2 → `fromSignedGrids` → (Task 4 not yet available) compare **raw** `signedDistance(ijk, l)` near each sphere: max interface error < 0.05·dx, RMS band error < 0.1·dx.
  - `RoundTripNonUnitVoxel`: same in index space at dx = 0.25 (radius 2.5, centres ±3.75); errors in dx units match the dx=1 run within 1e-3 (Review Focus 1).
  - `ConvergenceOrder`: sphere r = 0.5 world, dx ∈ {1/20, 1/40, 1/80}; least-squares slope of log(max interface error) vs log(dx) ≥ 1.8.
  - `FromLabelGrid`: a 20³ label block of 1 inside 30³ of 2 → `validate()`, `labelSet()=={1,2}`, φ at interface voxels within 0.5·dx ± 1e-3·dx of the cube faces.
  - `FromSignedGridsThrows`: input grid with class `GRID_FOG_VOLUME` → `ValueError`.
- [ ] **Step 2: Run `TestLevelSetMultiObject.RoundTrip*:*Convergence*:*From*`** — FAIL.
- [ ] **Step 3: Implement factories** per §3.1: union of input active topologies, dilated to halfWidth; label = argmin ψ (0 if all > 0); φ by the two-smallest rule; then `redistance()`. `fromLabelGrid` seeds φ = 0.5·dx at voxels with a different-label face neighbour, then redistances.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add MultiLevelSet factories with accuracy and convergence tests"`

---

### Task 4: Exact per-label extraction (§5.3, spec test 3)

**Files:** Modify `LevelSetMultiObject.h`; test in `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Consumes: `seedInterface` (Task 2).
- Produces: `typename FloatGridT::Ptr MultiLevelSet::extractSignedDistance(int32_t label, int halfWidth = -1) const;` (−1 → own halfWidth). Returns `GRID_LEVEL_SET`, background `halfWidth·dx`, same transform.

- [ ] **Step 1: Write failing tests.**
  - `TripleJunction`: three labels from half-spaces meeting at the z axis at 120° (label = sector of atan2(y, x)), 40³ domain. For each label, `extractSignedDistance` vs analytic signed distance to the sector: max interface error < 0.05·dx, RMS band < 0.1·dx. Also assert raw `signedDistance` underestimates somewhere outside each label by > 0.2·dx (regression guard).
  - `ExtractMatchesRoundTrip`: two-sphere fixture from Task 3 — extracted grid error ≤ raw error + 1e-4·dx.
  - `ExtractAbsentLabel`: label 99 → grid has 0 active voxels and background > 0 (Review Focus 2).
- [ ] **Step 2: Run `TestLevelSetMultiObject.TripleJunction:*Extract*`** — FAIL.
- [ ] **Step 3: Implement** per §5.3 steps 1–4. Seed with `IsInsideOp = L(ijk)==label`, then run the Task 2 Jacobi kernel on |ψ_l| (refactor it into a reusable function), then apply the sign.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add exact per-label signed distance extraction"`

---

### Task 5: `LabelAwareStencil` and `MultiObjectForces`

**Files:** Modify `LevelSetMultiObject.h`; test in `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces:
  ```cpp
  template<typename MultiLevelSetT>
  class LabelAwareStencil {   // reads main buffers (time n)
  public:
      explicit LabelAwareStencil(const MultiLevelSetT&);
      void moveTo(const Coord& ijk, int32_t label);
      template<int i, int j, int k> float getValue() const;   // offsets in [-1,1]
      Vec3f gradient() const;                       // central, world units
      void upwindDifferences(float dneg[3], float dpos[3]) const; // world units
      float meanCurvatureNormGrad() const;          // κ|∇ψ|, κ=(k1+k2)/2
  };
  // Speed functor concept (§3.3): operator()(ijk,label,stencil)->float, isFixed, isActive, stableDt(dx)
  template<typename MultiLevelSetT>
  class MultiObjectForces {
  public:
      void setCurvatureWeight(float); void setMaxCurvatureForce(float = 10.f);
      void setPressure(FloatGrid::ConstPtr, float weight, float target = std::numeric_limits<float>::quiet_NaN());
      void setAdvection(Vec3SGrid::ConstPtr, float weight);
      void setInactiveLabels(std::set<int32_t>); void setFixedLabels(std::set<int32_t>);
      float operator()(const Coord&, int32_t, const LabelAwareStencil<MultiLevelSetT>&) const;
      bool isFixed(int32_t) const; bool isActive(int32_t) const;
      float stableDt(float dx) const;               // dx²/(6·w) if w>0 else +inf
  };
  ```

- [ ] **Step 1: Write failing tests.**
  - `StencilSphere`: two-sphere fixture; at an interface voxel of sphere 1, `gradient()` has unit length ± 0.05 and points away from the centre; `meanCurvatureNormGrad()` within 10% of 1/r.
  - `StencilLabelSign`: at a label-2 voxel, `moveTo(ijk,1).getValue<0,0,0>() == +φ` and `moveTo(ijk,2)` gives `−φ`.
  - `ForcesPressureRescale`: pressure grid values {−2, 0, 4}, target 0 → internal values {−1, 0, 1}.
  - `ForcesMatchesWedgeFormula`: for one hand-built 3³ neighbourhood, `operator()` equals the wedge `pressureAndAdvectionMotion` expression (§3.4) evaluated by hand in the test, within 1e-6.
  - `ForcesFixedAndInactive`: label 0 → `isFixed` true; inactive label → `isActive` false.
  - `ForcesMissingGrid`: `setPressure(nullptr, 1.f)` or `setAdvection(nullptr, 1.f)` → `ValueError`.
- [ ] **Step 2: Run `TestLevelSetMultiObject.Stencil*:*Forces*`** — FAIL.
- [ ] **Step 3: Implement.** Stencil reuses `math::CurvatureStencil` arithmetic on a filled 19-value array (same index layout as `CurvatureStencil::mValues`). Advection samples with `BoxSampler` in index space of the velocity grid's transform.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add label-aware stencil and built-in multi-object forces"`

---

### Task 6: Evolver (§4, spec tests 4–7)

**Files:** Modify `LevelSetMultiObject.h`; test in `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Consumes: Tasks 1, 2, 5.
- Produces:
  ```cpp
  template<typename MultiLevelSetT, typename SpeedT, typename InterruptT = util::NullInterrupter>
  class LevelSetMultiObject {
  public:
      LevelSetMultiObject(MultiLevelSetT&, const SpeedT&, InterruptT* = nullptr);
      float  step(float maxDt);
      size_t advect(double t0, double t1);
      void setCFL(float); void setClampSpeed(bool); void setGrainSize(int); void setRedistanceTolerance(float);
  };
  ```
  Test helper `ConstantSpeed{float s; std::set<int32_t> fixed;}` satisfying the concept, plus `RandomSpeed` (per-label speed drawn once from a seeded `std::mt19937`, range [−1, 1]).

- [ ] **Step 1: Write failing tests.**
  - `PartitionInvariant` (spec test 4): 4 labels in a 32³ checker of 8³ blocks, `RandomSpeed`, 50 steps. After each step: `validate()`. At end: for each pair of labels, extracted interiors overlap at no voxel with both ψ < −1e-3·dx; every band voxel is inside exactly one label or label 0; no orphan voxel (every active voxel has a same-label voxel in its 26-neighbourhood).
  - `PressureGrowth`: sphere r = 8, constant pressure speed so that dr/dt = 1, advect to t = 4 → mean radius of extracted zero set 12 ± 0.5·dx.
  - `CurvatureFlow`: sphere r₀ = 10, curvature weight w = 1, t = 20 → r² = 100 − 2·20 = 60 ± (2·r·0.5·dx).
  - `UniformAdvection`: velocity (1,0,0), t = 5 → centroid shifts 5 ± 0.5·dx.
  - `FixedLabels` (spec test 6): label 2 fixed; all its voxels' (φ, L) bit-identical after 20 steps.
  - `BandMaintenance` (spec test 7): after every step of `PressureGrowth`, topology equal and max φ over active ≤ (halfWidth+1)·dx.
  - `EmptyStep`: pair with only label 0 → `step(0.5)` returns 0.5, trees unchanged (Review Focus 2).
  - `InterruptMidAdvect`: interrupter that returns `wasInterrupted()==true` on its 3rd call → `advect` returns < requested steps and `validate()` true (Review Focus 4).
  - `ThinFeatureVanishes`: a 1-voxel-thick slab of label 1 inside label 2, curvature + inward pressure, 30 steps → no NaN, `validate()` true, no orphan voxels (Review Focus 5).
- [ ] **Step 2: Run `TestLevelSetMultiObject.Partition*:*Growth:*Flow:*Advection:*Fixed*:*Band*:*EmptyStep:*Interrupt*:*Thin*`** — FAIL.
- [ ] **Step 3: Implement** §4 steps 1–6 exactly. Candidate list per voxel is a fixed `std::array<std::pair<int32_t,float>,7>` plus a count, in a per-leaf vector indexed by voxel offset. Swap aux→main on both trees in one function so they can't diverge.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add multi-object level set evolver"`

---

### Task 7: Single-object equivalence (spec test 8)

**Files:** Test only, `TestLevelSetMultiObject.cc`.

- [ ] **Step 1: Write tests** `SingleObjectAdvection`, `SingleObjectCurvature`, `SingleObjectCombined`. Fixture: sphere r = 12, dx = 1, label 1 vs background 0. Shared helper `compareZeroSets(const FloatGrid& a, const FloatGrid& b)` returns {symmetric-difference volume, Hausdorff of zero sets}. Hausdorff is computed from `volumeToMesh` points of each grid sampled into the other's `BoxSampler`.
  - Advection: rotational field v = (−y, x, 0)·0.05 as a `Vec3SGrid`; reference `LevelSetAdvection` with `FIRST_BIAS` / `TVD_RK1` on a copy of the sphere, advected to t = 20.
  - Curvature: reference = N = 60 iterations of `LevelSetFilter::meanCurvature()` (each dt = dx²/3, total t = 20); MOGAC curvature weight 1 to t = 20.
  - Combined: MOGAC advection + curvature to t = 10. The reference alternates `LevelSetAdvection` over dt = dx²/3 with one `meanCurvature()` iteration, 30 times.
  - Assert for each: symmetric difference < 1% of reference volume (`levelSetVolume`), Hausdorff < 1·dx.
- [ ] **Step 2: Run `TestLevelSetMultiObject.SingleObject*`** — expected PASS (the evolver exists). On failure, debug the evolver with superpowers:systematic-debugging; don't loosen thresholds.
- [ ] **Step 3: Commit** — `git commit -m "Test single-object MOGAC against single level set evolution"`

---

### Task 8: Eight objects in a tornado vortex (spec test 9)

**Files:** Test only, `TestLevelSetMultiObject.cc`.

- [ ] **Step 1: Write test `TornadoEightObjects`.** Domain index box [−64, 64]³, dx = 1.
  - Objects at radius 40 in the z = −20 plane, angles k·45°, labels 1–8, all of characteristic size 8. In order: `createLevelSetSphere`, `createLevelSetCube`, `createLevelSetTetrahedron`, `createLevelSetOctahedron`, `createLevelSetDodecahedron`, `createLevelSetIcosahedron`, `createLevelSetCapsule`, and a closed ring from `createLevelSetTubeComplex` (8-segment polygon, radius 6, tube radius 2).
  - Velocity `Vec3SGrid` over the domain with r = √(x²+y²) and core radius R = 16:
    - swirl u_θ = 1.0·(r/R) for r < R, else 1.0·(R/r);
    - inflow u_r = −0.3·exp(−(z+20)²/200);
    - updraft u_z = 0.4·exp(−r²/(2R²)).
  - Forces: advection weight 1, curvature weight 0.05. Advect to t = 120 in `advect` chunks of 10.
  - Every chunk: `validate()`, the partition check from Task 6, no NaN in φ.
  - At end:
    - distinct label pairs (a, b), a, b ≠ 0, sharing a face ≥ 4;
    - every label still owns ≥ 1 voxel;
    - per-label volume (voxel count with L == l) within ±25% of initial.
- [ ] **Step 2: Run `TestLevelSetMultiObject.TornadoEightObjects`** — PASS. If contact < 4, increase t to 200 (record the change in the test comment); don't change the thresholds.
- [ ] **Step 3: Commit** — `git commit -m "Add eight-object tornado vortex integration test"`

---

### Task 9: Connectivity-consistent case tables (§6.3 step 3, spec test 10)

**Files:**
- Create: `openvdb/openvdb/tools/LevelSetMultiObjectMesh.h`
- Create: `openvdb/openvdb/unittest/TestLevelSetMultiObjectMesh.cc`

**Interfaces:**
- Produces:
  ```cpp
  struct MultiObjectMeshSettings { enum class Connectivity { FG26_BG6, FG6_BG26, FG18_BG6, FG6_BG18 };
      Connectivity connectivity = Connectivity::FG26_BG6; bool junctionCorrection = true; bool computeNormals = false; };
  namespace lsmo_internal {
  struct CaseTriangulation { uint8_t count; std::array<std::array<uint8_t,3>,16> tris; }; // edge ids 0..11 (VolumeToMesh edge order)
  const std::array<CaseTriangulation,256>& caseTable(MultiObjectMeshSettings::Connectivity);
  }
  ```

- [ ] **Step 1: Write failing test `CaseTableAll`** — for each connectivity and each of the 256 cases:
  - every crossing edge (endpoints differ) is used, and no non-crossing edge;
  - every interior edge of the local triangle set has exactly 2 triangles, and boundary edges lie on cube faces;
  - winding: each triangle's normal (from a unit cube with midpoint vertices) has positive dot product with (centroid of out-corners − centroid of in-corners) of its component;
  - face consistency: the boundary segments on each face depend only on that face's 4 corner bits (compare against the generator's own face rule for all 16 face patterns);
  - components: the in-corners joined by the local surface regions equal the in-corner components under the cube-restricted fg adjacency (6: cube edges, 18: + face diagonals, 26: + body diagonals). Same for out-corners under bg adjacency.
- [ ] **Step 2: Run `TestLevelSetMultiObjectMesh.CaseTableAll`** — FAIL.
- [ ] **Step 3: Implement the generator** (runs once in a function-local static per connectivity):
  1. Per face, build crossing-edge segments. For an ambiguous face (diagonal in-pair), segments go around the out-corners (joining the in-pair) if fg ∈ {18, 26}, else around the in-corners.
  2. Chain segments into closed loops (each crossing edge lies on exactly 2 faces).
  3. Group loops by which in-component they bound. If two loops bound the same in-component under fg adjacency but are not one patch, which only happens for body-diagonal configurations under 26, connect them with a triangle-strip tube.
  4. Triangulate each remaining loop as a fan from its first vertex; orient by the winding rule above.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add connectivity-consistent case tables for multi-label contouring"`

---

### Task 10: `multiObjectToMesh` (§6.2–6.4, spec tests 11–13)

**Files:** Modify `LevelSetMultiObjectMesh.h`; test in `TestLevelSetMultiObjectMesh.cc`.

**Interfaces:**
- Consumes: `caseTable` (Task 9); `MultiLevelSet`, `seedInterface` (Tasks 1–2); fixtures from Tasks 3, 4, 8 (copy the fixture builders into a shared test header `openvdb/openvdb/unittest/LevelSetMultiObjectFixtures.h` in this task).
- Produces: `struct LabelMesh {...}` and `multiObjectToMesh(...)` exactly as §6.2.

- [ ] **Step 1: Write failing tests.** Helpers: `bool isClosedManifold(const LabelMesh&)` (edge→triangle count == 2, each vertex link one cycle, no zero-area triangles); `int components(const LabelMesh&)`; `int eulerChar(const LabelMesh&)`; `digitalTopology(labelGrid, l, conn) -> {fgComponents, cavities, chi}`. The last one uses a BFS with the chosen adjacency, and χ from the cubical complex (V − E + F − C over the closed unit cubes of the voxels, with the complex chosen per connectivity as in Lachaud–Montanvert).
  - `MeshManifold` (test 11): two spheres, tornado end state, and 5 random-noise label volumes (seeded, 24³, 4 labels): `isClosedManifold` for every label mesh.
  - `MeshTopology` (test 12), for each connectivity, assert components == fgComponents + cavities and χ(M) == 2·χ(R) on:
    - solid torus;
    - two cubes touching only along an edge: 1 component under fg 26 and 18, 2 under fg 6;
    - two cubes touching only at a corner: 1 component under fg 26, 2 under fg 18 and 6;
    - a hollow sphere;
    - the random volumes.
  - `MeshGeometrySphere` (test 13): sphere r = 10 — every vertex within 0.1·dx of the analytic surface.
  - `MeshJunctionCorrection` (test 13): triple-junction fixture — max vertex error with `junctionCorrection=true` < with `false`.
  - `MeshEmptyAndLabelIds`: background-only pair → empty map (Review Focus 2); labels −7 and 1<<30 → both keys present with closed meshes (Review Focus 3).
- [ ] **Step 2: Run `TestLevelSetMultiObjectMesh.Mesh*`** — FAIL.
- [ ] **Step 3: Implement** §6.3 steps 1–5 and §6.4. Edge key = (min-corner `Coord`, axis, label) hashed into leaf-local `std::unordered_map`. Merge across leaves in a serial pass over boundary edges, keyed globally.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add topology-constrained multi-label contouring"`

---

### Task 11: Benchmark, docs, pending changes, full run (spec test 14)

**Files:**
- Test: `TestLevelSetMultiObjectMesh.cc`
- Modify: both headers (Doxygen)
- Create: `pendingchanges/multi_object_level_set.txt`

- [ ] **Step 1: Write `MeshBenchmark`** — the tornado end state (fixture header). Time `multiObjectToMesh` (all labels) against the sum over labels of `extractSignedDistance` + `volumeToMesh`, best of 3 runs each, and print both. `EXPECT_LE(multiTime, perLabelTime)`.
- [ ] **Step 2: Run `TestLevelSetMultiObjectMesh.MeshBenchmark`** — PASS, with timings printed.
- [ ] **Step 3: Doxygen.**
  - File-level `@brief`, usage example (build pair → forces → `advect` → `extractSignedDistance` → `multiObjectToMesh`) and the §10 references.
  - Note in the MOGAC/MUSCLE citation that venue and year are pending the author.
- [ ] **Step 4: Pending changes file** in the existing format:
  ```
  OpenVDB:
    New Features:
    - Added tools::LevelSetMultiObject (multi-object geodesic active contours on a paired unsigned-distance/label grid), with second-order per-label signed distance extraction and topology-constrained multi-label contouring (tools::multiObjectToMesh).
  ```
- [ ] **Step 5: Full run** — `build/openvdb/openvdb/unittest/vdb_test --gtest_filter='TestLevelSetMultiObject*'`. Expected: all pass. Then reconfigure with `-DOPENVDB_TESTS=""` plus `-DOPENVDB_CXX_STRICT=ON` and build `vdb_test` to confirm no warnings in other translation units that include the headers.
- [ ] **Step 6: Commit** — `git commit -m "Document multi-object level set tools and add benchmark"`
