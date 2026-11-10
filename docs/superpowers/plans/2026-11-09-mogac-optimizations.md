# MOGAC Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MOGAC tool faster (splitting, incremental redistance, aux buffers, direct field reads, optional full-voxel steps), more accurate (HJWENO5 + TVD-RK2, speed extension, closest-point junctions on by default, optional volume constraint) and leaner (16-bit labels by default, pooled scratch).

**Architecture:** All changes stay in `tools/LevelSetMultiObject.h` (holder, redistancer, stencil, forces, evolver) and `tools/LevelSetMultiObjectMesh.h` (junction option). Refactors that must not change results come first and are proven bit-identical against the current suite; behavior changes follow, each behind a setter with a documented default.

**Tech Stack:** C++17, OpenVDB (`tree::LeafManager`, `math::` HJWENO5/TVD schemes, `BoxSampler`), TBB, GoogleTest, CMake.

**Spec:** `docs/superpowers/specs/2026-11-09-mogac-optimizations-design.md` (addendum) on top of `docs/superpowers/specs/2026-11-08-multi-object-level-set-design.md`.

## Global Constraints

- Branch `feature/mogac-optimizations`; work in `/home/blake/workspace/openvdb`; never touch `/home/blake/workspace/openvdb-mogac-renders`.
- Header-only; namespace `openvdb::OPENVDB_VERSION_NAME::tools`, internals in `tools::lsmo_internal`; errors via `OPENVDB_THROW(ValueError, …)`; asserts via `OPENVDB_ASSERT`; no existing OpenVDB header modified; lines ≤ 100 columns in the two tool headers; no references to rulings/spec/wedge in public header docs.
- World units for φ, ψ, speeds; dt in world time; uniform voxel size.
- TBB with settable grain size; ≤ 0 runs serially; results independent of grain size.
- Defaults after this plan: spatial scheme `math::HJWENO5_BIAS`, temporal `math::TVD_RK2`, splitting on for split functors, incremental redistance on, max interface motion 0.5 voxel (1.0 only if Task 10's gates pass), speed extension on, closest-point junctions on, volume constraints off, label tree `Int16LabelTree`.
- Existing tests: `SingleObject*` pin `FIRST_BIAS`/`TVD_RK1` (they compare against OpenVDB's first-order reference). All other existing tests run with the defaults and keep their current gates; a gate may change only by controller ruling.
- Build: `cmake --build build --target vdb_test -j14 && build/openvdb/openvdb/unittest/vdb_test --gtest_filter='TestLevelSetMultiObject*'` (60 tests today, ~44 s). Strict build in `build-strict/` at Task 11.

## Review Focus

1. Loading a wide-label grid (Int32) with a label outside ±32767 into the 16-bit default → `ValueError`, not silent truncation. Pinned in Task 4.
2. TVD-RK2 stages with topology protection, fixed walls and passive label 0 → protection holds at every stage (TopologyCubeThroughSpheres and FixedLabels pass with defaults). Pinned in Task 6.
3. Splitting with fixed labels and passive background → fixed voxels bit-identical, growth at full speed. Pinned in Task 5.
4. Volume constraint on a label that vanishes or has zero interface area → no NaN/inf, constraint skipped for that step. Pinned in Task 9.
5. Velocity/pressure grids whose transform differs from the pair (offset or scale) → sampler path still used and results unchanged. Pinned in Task 2.

---

### Task 1: Aux buffers and pooled scratch (S4, M3, M2)

**Files:** Modify `openvdb/openvdb/tools/LevelSetMultiObject.h` (evolver `step()` / `swapBuffers` / scratch); Test `openvdb/openvdb/unittest/TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces (evolver, private): per-step φ/label double buffering through one `tree::LeafManager` aux buffer per tree, swapped together; a `Scratch` member owning per-leaf candidate arrays and BoolTree masks, reused across steps and resized only when the leaf count grows. No public API change.

- [ ] **Step 1: Write failing test `StepPeakMemory`:** a 64³-band sphere pair; record `phi()->memUsage() + labels()->memUsage()` before a step and the evolver's reported scratch/aux bytes via a new `size_t LevelSetMultiObject::scratchMemUsage() const`; assert `scratchMemUsage() ≤ 1.3 × pairBytes` (deep copies today make it ≥ 2×).
- [ ] **Step 2: Run** `--gtest_filter='TestLevelSetMultiObject.StepPeakMemory'` — FAIL (method missing).
- [ ] **Step 3: Implement** aux-buffer double buffering and the pooled scratch; add `scratchMemUsage()`.
- [ ] **Step 4: Run all** `TestLevelSetMultiObject*` — all pass; additionally assert in the report that every printed numeric value of the existing tests is unchanged (diff the test output against the base commit, timing lines excluded).
- [ ] **Step 5: Commit** — `git commit -m "MOGAC evolver: aux buffers and pooled scratch instead of per-step deep copies"`

---

### Task 2: Direct field reads (S5)

**Files:** Modify `LevelSetMultiObject.h` (`MultiObjectForces` sampling, `LabelAwareStencil` field cache); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces: `MultiObjectForces` reads a velocity/pressure grid by direct accessor `getValue(ijk)` when `grid.transform() == pair transform` (index spaces coincide), else the existing `BoxSampler` path. No API change.

- [ ] **Step 1: Write failing test `FieldDirectReadMatchesSampler`:** run UniformAdvection-like and pressure-growth-like scenarios twice — with the field grid on the pair's transform and with a copy whose transform is translated by (0.5, 0.25, 0)·dx and resampled — assert the aligned run's result is bit-identical to running with direct reads disabled via a test-only `MultiObjectForces::setForceSampling(true)`, and the misaligned run still matches its own sampler path (Review Focus 5).
- [ ] **Step 2: Run** — FAIL (setter missing).
- [ ] **Step 3: Implement** the aligned-transform fast path and `setForceSampling(bool)` (default false).
- [ ] **Step 4: Run all** — pass, outputs unchanged.
- [ ] **Step 5: Commit** — `"MOGAC forces: direct field reads on aligned transforms"`

---

### Task 12: Topology debug utilities + orbiting spheres in the cube test (user requests) — run after Task 2, before Task 3

**Files:** Modify `openvdb/openvdb/tools/LevelSetMultiObject.h` (per-label advection in `MultiObjectForces`), `openvdb/openvdb/unittest/LevelSetMultiObjectTopology.h` (debug utilities), `TestLevelSetMultiObject.cc`, `TestLevelSetMultiObjectMesh.cc`.

**Interfaces:**
- Produces (library): `void MultiObjectForces::setAdvection(int32_t label, Vec3SGrid::ConstPtr field, float weight)` — overrides the shared advection field for that label (null field throws `ValueError`); `void clearLabelAdvection(int32_t label)`. Labels without an override use the shared field (behavior unchanged).
- Produces (test-only, `lsmo_test` namespace): `struct MeshTopology { int components; int euler; bool closed; bool manifold; std::vector<Vec3s> boundaryEdgeMidpoints, nonManifoldEdgeMidpoints; }; MeshTopology meshTopology(const tools::LabelMesh&);` `struct TopologyReport { bool ok; std::string message; DigitalTopology region; MeshTopology mesh; };` `template<typename MLS> TopologyReport verifyLabelTopology(const MLS&, int32_t label, const CoordBBox& domain, tools::MultiObjectConnectivity, const std::string& tag, const DigitalTopology* expected = nullptr);` — checks mesh components == region components + cavities, χ(mesh) == 2·χ(region), closed and manifold, and `expected` if given; macro `EXPECT_TOPOLOGY_CONSISTENT(report)` (EXPECT_TRUE(report.ok) << report.message). Debug dump on failure, or always when env `OPENVDB_MOGAC_DEBUG_DIR` is set: `<dir>/<gtest test name>/<tag>/` gets `pair.vdb` (φ + labels, int16 grids registered via the existing helper if present, else written as Int32 copy), `label<L>.ply`, `report.txt` (per-component voxel counts and bboxes, cavity bboxes, boundary/non-manifold edge positions). `digitalTopology` is templated on the label grid type (O2). Connectivity rule (user requirement): the region's topology is always computed with the digital connectivity pair the object is evolved/meshed under — foreground components with the pair's fg adjacency (6/18/26), cavities with its bg adjacency, Euler characteristic with the matching cubical-complex convention — and `verifyLabelTopology` uses that single `MultiObjectConnectivity` for the region, the mesh (`multiObjectToMesh` settings) and the expectation. Add `template<typename EvolverT> tools::MultiObjectConnectivity connectivityOf(const EvolverT&)` (returns the evolver's `setConnectivity` value) so tests pass the evolver's rule rather than a literal; no test hard-codes a different rule for verification than for evolution.

- [ ] **Step 1: Write failing tests:**
  - `MeshTopologyUtility`: meshTopology on a closed sphere mesh (1 component, χ 2, closed, manifold) and on a mesh with one triangle removed (not closed, 3 boundary edges reported); verifyLabelTopology on the existing torus/hollow-sphere fixtures returns ok with the expected numbers; a deliberately wrong `expected` makes ok false, the message names the mismatch, and with OPENVDB_MOGAC_DEBUG_DIR set the three dump files exist.
  - `PerLabelAdvection`: two spheres, label 1 with the shared field (1,0,0), label 2 with override (0,1,0): after t = 5 the centroids moved by (5,0,0) and (0,5,0) ± 0.5·dx; null override throws.
  - `ConnectivityRuleMatters`: a fixture whose digital topology differs between rules (two blocks touching only at a corner: 1 component under FG26_BG6, 2 under FG6_BG26); `verifyLabelTopology` with each rule reports that rule's numbers and is consistent with that rule's mesh; verifying with the wrong rule against the other rule's expectation fails with a message naming the rule.
  - Rewrite `TopologyCubeThroughSpheres` (and its control), parameterized over the protection rules FG26_BG6 and FG6_BG26 (evolver `setConnectivity(rule)`, verification via `connectivityOf(evolver)`): the three spheres (radius 4, same initial positions) are ordinary labels (no longer fixed) advected by a per-label rigid rotation about the axis parallel to x through their common centroid, in the plane orthogonal to the cube's motion, with angular speed ω = 2π / 84 (one revolution over the run; tangential speed ≤ 0.25·dx per unit time at the spheres' radius); the cube (label 1, protected, FG26_BG6) keeps velocity (1,0,0) and travel 84. Every chunk: `EXPECT_TOPOLOGY_CONSISTENT(verifyLabelTopology(..., expected (1,0,1)))` for the cube; `validate()`, partition invariant; each sphere stays one component under the same rule. End: cube leading face moved 84 ± 2·dx; each sphere centroid within 2·dx of its analytically rotated position; sphere volume loss ≤ 1.1 × OpenVDB `LevelSetAdvection` (FIRST_BIAS/TVD_RK1) loss for the same sphere in the same rotation over the same time + 2%. Control (cube unprotected): its topology changes at some chunk (report via verifyLabelTopology with dump), as before.
  - Replace the inline topology checks in the mesh topology tests and DigitalTopologyHelper with `verifyLabelTopology` / `meshTopology` (no duplicated logic).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** `TestLevelSetMultiObject*` — pass.
- [ ] **Step 5: Commit** — `"MOGAC: topology debug utilities, per-label advection, orbiting spheres in the cube test"`

---

### Task 3: Incremental redistancing (S2)

**Files:** Modify `LevelSetMultiObject.h` (`MultiLevelSetRedistance`: new `redistance(const BoolTreeT& freeze, const BoolTreeT& dirty)` overload; evolver builds the dirty mask); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Consumes: Task 1 scratch masks.
- Produces: `int MultiLevelSetRedistance::redistance(const BoolTreeT& freeze, const BoolTreeT& dirty)` — solves only active, non-frozen voxels in `dirty`; others keep values. Evolver: `void setFullRedistance(bool)` (default false = incremental). Dirty mask = dilate(projected ∪ label-changed ∪ newly activated, halfWidth) ∩ band. Warm start per spec §S2.

- [ ] **Step 1: Write failing test `IncrementalRedistanceMatchesFull`:** for PressureGrowth, UniformAdvection, CurvatureFlow, Tornado and TopologyCubeThroughSpheres scenarios, run with `setFullRedistance(true)` and `(false)`; assert max |φ_full − φ_incr| ≤ 1e-4·dx over active voxels and identical label trees after every chunk.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass; report per-step time before/after on the Tornado scenario.
- [ ] **Step 5: Commit** — `"MOGAC: incremental redistancing over the dirty region"`

---

### Task 13: API cleanup — remove the clamp-speed option; bool-only validators (user requests) — run after Task 3

**Files:** Modify `openvdb/openvdb/tools/LevelSetMultiObject.h` (evolver), `docs/superpowers/specs/2026-11-08-multi-object-level-set-design.md` (§3.5 setters, §4 step 3).

**Interfaces:**
- Removes: `LevelSetMultiObject::setClampSpeed(bool)`, `clampSpeed() const`, member `mClampSpeed`, the speed clamp to [−1, 1] and the `dt = min(dt, 0.5·dx)` clamp branch. dt is always min(maxDt, motion cap / max|F|, stableDt). Docs mentioning the option are removed. The curvature-term clamp in `MultiObjectForces` (`setMaxCurvatureForce`) is a different mechanism and stays unless the user says otherwise.

- [ ] **Step 0 (user request): bool-only validators.** `MultiLevelSet::validateBackgrounds()` and `validatePhi()` take no arguments and return only `bool` (no message parameter, no message building). Callers produce the text: the constructor throws `ValueError` with a fixed message per failed check ("phi background must equal halfWidth * voxel size and the label background must be 0" / "phi must be non-negative and finite at every active voxel"); `validate(std::string* message)` keeps its signature and sets the same fixed texts (plus its existing topology-mismatch text). Update tests that asserted on detailed per-voxel message contents to assert on the fixed texts.
- [ ] **Step 1: Write failing test `NoClampSpeedOption`:** a compile-time check (`static_assert` via a detection trait) that `LevelSetMultiObject<…>` has no `setClampSpeed` member.
- [ ] **Step 2: Run** — FAIL (member exists).
- [ ] **Step 3: Remove** the option, branches and docs; update the base spec text.
- [ ] **Step 4: Run all** `TestLevelSetMultiObject*` — pass with unchanged numeric outputs (no test used the option).
- [ ] **Step 5: Commit** — `"MOGAC: remove clamp-speed option; bool-only validators"`

---

### Task 14: Style pass — American spelling and named constants (user requests) — run after Task 13

**Files:** `openvdb/openvdb/tools/LevelSetMultiObject.h`, `openvdb/openvdb/tools/LevelSetMultiObjectMesh.h`, `openvdb/openvdb/unittest/TestLevelSetMultiObject.cc`, `TestLevelSetMultiObjectMesh.cc`, `LevelSetMultiObjectFixtures.h`, `LevelSetMultiObjectTopology.h`, `pendingchanges/multi_object_level_set.txt`, `docs/superpowers/specs/2026-11-0*-*.md`, `docs/superpowers/plans/2026-11-0*-*.md`.

**Interfaces:** identifier renames only, e.g. `coloredPasses` → `coloredPasses`, `byColor` → `byColor`, any `*Neighbor*`/`*Center*`/`*Behavior*` → American forms. No behavior (behavior) change.

- [ ] **Step 1:** Replace British spellings with American ones in identifiers, comments, docs, test names and messages: the British forms c-o-l-o-u-r (and -ed), n-e-i-g-h-b-o-u-r (all forms), c-e-n-t-r-e (-s, -d), b-e-h-a-v-i-o-u-r, doubled-l labe-ll-ed / mode-ll-ed / trave-ll-ed, g-r-e-y, f-a-v-o-u-r, and British -ise/-ised/-ising/-isation endings (initialise, normalise, maximise, analyse, parameterise) become color, neighbor, center, behavior, labeled, modeled, traveled, gray, favor and -ize/-ized/-izing/-ization. Words that are -ise in American English stay: otherwise, clockwise, noise, raise, exercise, pairwise, piecewise, bitwise. (Spelled with hyphens here so this plan itself passes the check.) Do not rename identifiers in OpenVDB code outside the feature's files. Check every gtest name still matches what CI filters reference (none reference renamed tests outside these files).
- [ ] **Step 1b: Named constants (user request).** Replace magic numbers in the two tool headers with named `constexpr` symbols in `tools::lsmo_internal` (or the owning class), each with a one-line comment stating its meaning and units, e.g. `kBarrierPhiFraction = 1e-3` (the "1e-3" in `std::max(1e-3 * dx, this->phiFloor())`, the protection barrier and kinematic-stamp minimum φ, as a fraction of dx), `kDefaultMinimumPhiFraction`, `kSeedEpsilonFraction` (1e-6·dx guards), `kMaxInterfaceMotionVoxels` (0.5), `kGradientNormSqEpsilon` (1e-8), the inactive-label speed (0.999), θ clamps (1e-3 / 1 − 1e-3), default CFL, default redistance tolerance fraction (1e-4), default max curvature force (10), default volume relaxation once it exists, etc. Every occurrence of the same meaning uses the same symbol; numerically identical literals with different meanings get different symbols. Leave literals that are self-explanatory structural values (0, 1, 2, 3 as indices/dimensions, 0.5 in a midpoint average, the 27 colors) alone. Existing named constants (`kMaxWrapPasses`, `kMeshMinQuality`) stay; move them next to the others if that improves discoverability.
- [ ] **Step 2: Verify** a case-insensitive grep over the files above for the British forms listed in Step 1 (e.g. the regex `colo[u]r|neighbo[u]r|cent[r]e\b|behavio[u]r|labe[l]led|mode[l]led|trave[l]led|\bgr[e]y\b|favo[u]r|initiali[s]|normali[s]|maximi[s]|minimi[s]|analy[s]e|parameteri[s]`) returns nothing.
- [ ] **Step 3: Run all** `TestLevelSetMultiObject*` — pass with unchanged numeric outputs; strict build of the two test files clean.
- [ ] **Step 4: Commit** — `"MOGAC: American spelling and named constants"` (two commits are fine)

---

### Task 15: Hashed containers on hot paths (user request) — run after Task 14

**Files:** `openvdb/openvdb/tools/LevelSetMultiObject.h`, `openvdb/openvdb/tools/LevelSetMultiObjectMesh.h`, test helper headers if they hold hot lookups.

**Interfaces:** `lsmo_internal::CoordHash` (hash for `Coord`, e.g. from its three components; usable with std::unordered_map/set). Internal containers switch to hashed versions: `mInactive`, `mFixed`, `mProtected` (per-candidate lookups) → `std::unordered_set<int32_t>` (public setters keep taking `std::set<int32_t>` and convert; getters that return `const std::set&` keep an ordered copy or change to return by value — choose and document); `mLabelAdvection`, `mKinematic` → `std::unordered_map<int32_t, …>` where only looked up, keeping any order-dependent iteration deterministic; per-step `std::map<Coord, size_t>` leaf index, `claimObject`, wrap `adds`, `claimed` → `std::unordered_map/set<Coord, …, CoordHash>`. Wherever iteration order affects results (serial apply of deferred changes, wrap adds, claim order, first-object-wins), iterate over a sorted vector of keys or keep the existing ordered container — results must stay identical and thread-count independent. Public API containers (fromSignedGrids input map, multiObjectToMesh result map, label-set setters) stay ordered.

- [ ] **Step 1: Write test `HashedContainersKeepResults`:** run the Tornado, TopologyCubeThroughSpheres (both rules, kinematic spheres) and PartitionInvariant scenarios and compare φ/label tree hashes against values recorded from the pre-change build (record them first, commit them as constants in the test); also grain 0 vs 64 identical.
- [ ] **Step 2: Implement** the container changes.
- [ ] **Step 3: Run all** `TestLevelSetMultiObject*` — pass; numeric outputs identical; report per-step timing before/after on Tornado and the cube test (keep a change only where it is not slower — revert any hashed container that measures slower and note it).
- [ ] **Step 4: Commit** — `"MOGAC: hashed containers on hot paths"`

---

### Task 16: Rename Tornado → Vortex; double the vortex run (user request) — run after Task 15

**Files:** `openvdb/openvdb/unittest/TestLevelSetMultiObject.cc`, `TestLevelSetMultiObjectMesh.cc`, `LevelSetMultiObjectFixtures.h`, any tool-header doc that names the scenario, `docs/superpowers/specs/*.md`, `docs/superpowers/plans/*.md`.

**Interfaces:** rename every "Tornado"/"tornado" identifier, test name, fixture struct, helper, scenario label and doc mention to "Vortex"/"vortex" (e.g. `TornadoEightObjects` → `VortexEightObjects`, `lsmo_test::Tornado` → `lsmo_test::Vortex`). Double the vortex simulation length: end time t = 42 → 84 (keep the chunk size; the "two chunks past first contact" rule is replaced by running to t = 84 with contacts ≥ 4 required at the end and at every chunk after first contact). All gates unchanged in form: per-chunk validate/partition/no-NaN; ≥ 4 contact pairs; every label survives; two-sided per-label volume gate vs OpenVDB LevelSetAdvection over the same (doubled) time. Scenarios that reuse the vortex at full length (IncrementalRedistanceMatchesFull's asserted runs, HashedContainersKeepResults, StepBenchmark/MeshBenchmark end state) follow the new length; re-record any committed hashes from the pre-change build at the new length and say so.

- [ ] **Step 1:** Rename (verify `grep -rni tornado` over the listed files returns nothing).
- [ ] **Step 2:** Double the vortex run and update dependents.
- [ ] **Step 3: Run all** `TestLevelSetMultiObject*` — pass. If a vortex gate fails at t = 84 (lost contacts, a vanished label, volume gate), STOP and report the numbers per label/chunk instead of changing a gate. Report runtime before/after.
- [ ] **Step 4: Commit** — `"MOGAC tests: rename Tornado to Vortex; run the vortex twice as long"`

---

### Task 4: 16-bit labels by default (M1)

**Files:** Modify `LevelSetMultiObject.h` (`Int16LabelTree`, default template argument, `registerMultiLevelSetGridTypes()`, factory range checks), `LevelSetMultiObjectMesh.h` if it names `Int32Tree`; Tests in both test files and `LevelSetMultiObjectFixtures.h`.

**Interfaces:**
- Produces: `using Int16LabelTree = tree::Tree4<int16_t, 5, 4, 3>::Type;`, `using Int16LabelGrid = Grid<Int16LabelTree>;`, `template<typename FloatTreeT = FloatTree, typename LabelTreeT = Int16LabelTree> class MultiLevelSet`, `inline void registerMultiLevelSetGridTypes();` (registers `Int16LabelGrid` if not registered; idempotent). `fromLabelGrid` accepts any integer label grid type (template on the input grid) and throws `ValueError` when a label is outside the target label type's range.

- [ ] **Step 1: Write failing tests:** `Int16LabelsDefault` (`static_assert` default label tree is `Int16LabelTree`; a two-sphere pair's `labels()->memUsage()` < the Int32 variant's), `Int16RangeThrows` (Int32 label grid with label 40000 → `ValueError`; label −32768 accepted) (Review Focus 1), `Int16GridIO` (`registerMultiLevelSetGridTypes()`, write φ+labels to a temp .vdb with `io::File`, read back, `validate()` and labels identical). Switch tests that use labels like `1<<30` to an explicit `MultiLevelSet<FloatTree, Int32Tree>`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass with unchanged numeric outputs (labels small in all scenarios).
- [ ] **Step 5: Commit** — `"MOGAC: 16-bit labels by default; register int16 label grids for I/O"`

---

### Task 5: Operator splitting (S1)

**Files:** Modify `LevelSetMultiObject.h` (speed concept trait, `MultiObjectForces::hyperbolic/parabolic`, evolver Strang step, `setSplitting`); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces: in `MultiObjectForces`: `float hyperbolic(const Coord&, int32_t, const StencilT&) const` (advection + pressure + inactive-label speed), `float parabolic(const Coord&, int32_t, const StencilT&) const` (curvature term incl. interface correction and clamp), `float parabolicStableDt(float dx) const` (= dx²/(6w) or +inf); `operator()` = hyperbolic + parabolic. `lsmo_internal::IsSplitSpeed<SpeedT>` trait (true when both members exist). Evolver: `void setSplitting(bool)` (default true; ignored for non-split functors). Strang step: parabolic(dt/2, subcycled to parabolicStableDt) → hyperbolic(dt, motion cap) → parabolic(dt/2); each sub-update is a full MOGAC update (projection, protection, incremental redistance, band maintenance).

- [ ] **Step 1: Write failing tests:** restore `SingleObjectCombined` to t = 15 (45 reference iterations, rotation 0.05·15, r² = r₀² − 2·15) with its existing gate, splitting on; add `SplittingFixedAndPassive` (FixedLabels scenario with curvature + pressure: fixed voxels bit-identical; PressureGrowth with a curvature weight of 0.01 grows within its gate) (Review Focus 3).
- [ ] **Step 2: Run** — Combined FAILS at t = 15 without splitting (0.619 vs 0.605), new test FAILS (API).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass; report Combined radius error at t = 10 and 15 and step counts before/after.
- [ ] **Step 5: Commit** — `"MOGAC: Strang splitting of advection/pressure and curvature"`

---

### Task 6: HJWENO5 + TVD-RK2 (A1)

**Files:** Modify `LevelSetMultiObject.h` (stencil radius 3, upwind HJWENO5 via `math::` operators, RK2 stages with per-stage projection, setters, defaults); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces: evolver `void setSpatialScheme(math::BiasedGradientScheme)` (FIRST_BIAS or HJWENO5_BIAS, default HJWENO5_BIAS; others throw `ValueError`), `void setTemporalScheme(math::TemporalIntegrationScheme)` (TVD_RK1 or TVD_RK2, default TVD_RK2); `LabelAwareStencil` gains radius-3 axis reads and `void upwindDifferencesWeno5(float dneg[3], float dpos[3]) const`. Requires halfWidth ≥ 3 for HJWENO5 (throw at construction/setter). `MultiObjectForces` uses the evolver-selected upwind differences (passed via the stencil).

- [ ] **Step 1: Write failing tests:** pin `SingleObject*` to FIRST_BIAS/TVD_RK1 explicitly; `Weno5ReducesAdvectionError` (SingleObjectAdvection scenario: HJWENO5/RK2 symdiff vs analytic < 0.6 × FIRST_BIAS/RK1's); `Weno5ReducesTornadoLoss` (Tornado with defaults: worst per-label volume loss < 0.6 × the FIRST_BIAS/RK1 run's); `Rk2StagesRespectProtection` = TopologyCubeThroughSpheres and FixedLabels pass with defaults (Review Focus 2); `Weno5NeedsHalfWidth3` (halfWidth 2 + HJWENO5 → `ValueError`).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass; non-SingleObject tests keep their gates with the new defaults (report any gate that needed a controller ruling instead of changing it).
- [ ] **Step 5: Commit** — `"MOGAC: HJWENO5 upwinding and TVD-RK2 with per-stage projection"`

---

### Task 7: Speed extension (A2)

**Files:** Modify `LevelSetMultiObject.h` (`MultiObjectForces` field sampling point); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces: `MultiObjectForces::setSpeedExtension(bool)` (default true): advection velocity and pressure are sampled at x_Γ = x − ψ·∇ψ/|∇ψ|² (world), falling back to x when |∇ψ|² < 1e-8; one sample per voxel reused across candidates and RK stages (Task 2/6 cache).

- [ ] **Step 1: Write failing tests:** `SpeedExtensionKeepsDistance` — ExtractAccuracyAfterEvolution's advection case in a shear velocity field (v = (0.05·z, 0, 0)) with extension on vs off: interface max error on ≤ off, both ≤ 0.1·dx; CurvatureFlow and ExtractAccuracyAfterEvolution keep their gates.
- [ ] **Step 2: Run** — FAIL (setter missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass.
- [ ] **Step 5: Commit** — `"MOGAC forces: sample fields at the closest interface point"`

---

### Task 8: Closest-point junction reconstruction, on by default (A3)

**Files:** Modify `LevelSetMultiObject.h` (`ExtractSettings`, extraction overload, `lsmo_internal::closestPointPsi`), `LevelSetMultiObjectMesh.h` (`MultiObjectMeshSettings::closestPointJunctions`); Tests in both test files.

**Interfaces:**
- Produces: `struct ExtractSettings { int halfWidth = -1; int grainSize = 1; bool closestPointJunctions = true; };` and `template<typename InterruptT = util::NullInterrupter> typename FloatGridT::Ptr extractSignedDistance(int32_t label, const ExtractSettings&, InterruptT* = nullptr) const;` (the existing overload forwards to it). `lsmo_internal::closestPointPsi(const FloatTreeT& phi, const LabelTreeT& labels, const Coord& q, int32_t label, double dx) -> float` — ψ_l(q) from local planes through p − φ_p·n_p (n_p from one-sided ψ_l differences inside l) over l-side seeds in q's 5³ neighborhood, clamped to [φ_q, min_p(|q−p| − φ_p)]. `MultiObjectMeshSettings::closestPointJunctions = true` (on by default, user decision).

- [ ] **Step 1: Write failing tests:** `ClosestPointJunctions` over the existing 12-junction set: with the option on, worst interface error within 2 voxels of a junction line < the option-off worst (0.377·dx today) and far-field (≥ 2 voxels) ≤ 0.05·dx; off reproduces today's numbers exactly. Existing TripleJunction and mesh junction tests now run with the option on (default) and keep their gates. `MeshClosestPointJunctions`: mesher worst junction vertex error with the option < without; topology checks unchanged.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass.
- [ ] **Step 5: Commit** — `"MOGAC: closest-point reconstruction at triple junctions (default on)"`

---

### Task 9: Optional per-label volume constraint (A4)

**Files:** Modify `LevelSetMultiObject.h` (evolver volume bookkeeping, λ_l added to the hyperbolic speed); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Produces: evolver `void setVolumeConstraint(int32_t label, double targetVolume)`, `void preserveInitialVolume(int32_t label)`, `void clearVolumeConstraints()`, `void setVolumeRelaxation(double)` (default 0.5), `double labelVolume(int32_t) const` (voxel count·dx³ + sub-voxel interface correction from φ). Per step: λ_l = relaxation·(V_l − V_target)/(A_l·dt), A_l from the interface voxels' |∇ψ|·dx² sum; skipped (λ_l = 0) when A_l·dt ≤ 1e-12 or V_l = 0.

- [ ] **Step 1: Write failing tests:** `VolumeConstraintTornado` — Tornado with every label `preserveInitialVolume`: each label's volume within 2% of initial at the end, contacts ≥ 4; unconstrained labels in a mixed run behave as before. `VolumeConstraintVanishingLabel` — a label that vanishes under strong inward pressure with a constraint on another label: no NaN/inf, `validate()` holds (Review Focus 4).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all** — pass.
- [ ] **Step 5: Commit** — `"MOGAC: optional per-label volume constraint"`

---

### Task 10: Full-voxel interface motion (S3)

**Files:** Modify `LevelSetMultiObject.h` (evolver projected set, ring-2 candidates via `closestPointPsi`); Test `TestLevelSetMultiObject.cc`.

**Interfaces:**
- Consumes: `lsmo_internal::closestPointPsi` (Task 8).
- Produces: evolver `void setMaxInterfaceMotion(double voxels)` (0.5 or 1.0; other values throw `ValueError`). At 1.0: projected set = φ ≤ 1.5·dx or within one voxel of a different label; ring-2 candidates from the 26-neighborhood with ψ_l = `closestPointPsi`.

- [ ] **Step 1: Write failing tests:** `FullVoxelMotion` — run ContactNoCascade (probe geometry: label-2 extent growth ≤ 1 voxel per step), PressureGrowth, UniformAdvection, CurvatureFlow, Tornado, TopologyCubeThroughSpheres and FixedLabels at motion 1.0 with their existing gates; also assert the step count is ≤ 0.6 × the 0.5 run's for PressureGrowth.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all.** If every gate passes at 1.0, make 1.0 the default and re-run everything; otherwise keep 0.5 as the default, keep `FullVoxelMotion` asserting only the gates that pass, and report the failing ones with numbers (controller rules).
- [ ] **Step 5: Commit** — `"MOGAC: optional full-voxel interface motion per step"`

---

### Task 11: Benchmark, docs, strict build

**Files:** Modify both tool headers (Doxygen for every new setter/default), `pendingchanges/multi_object_level_set.txt`; Test `TestLevelSetMultiObject.cc` (`StepBenchmark`).

- [ ] **Step 1: Write `StepBenchmark`** (informational, prints only): Tornado and a ~1M-voxel sphere-growth scene; per-step time and steps for t = 10 with (a) FIRST_BIAS/RK1/no splitting/full redistance/0.5 motion — the pre-plan configuration — and (b) defaults; print both and the speedup.
- [ ] **Step 2: Docs:** document every new setter, the new defaults, `registerMultiLevelSetGridTypes()` and the int16 range; update pendingchanges (wrapped like the other entries).
- [ ] **Step 3: Run all** `TestLevelSetMultiObject*` in `build/`; rebuild `build-strict/` (OPENVDB_CXX_STRICT=ON) in the background and run the same filter there — no warnings.
- [ ] **Step 4: Commit** — `"MOGAC optimizations: benchmark, docs, pending changes"`
