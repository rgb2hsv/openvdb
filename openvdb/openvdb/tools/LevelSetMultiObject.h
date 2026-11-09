// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObject.h
///
/// @brief Multi-object level sets: a single unsigned distance grid phi
/// paired with an integer grid of object labels (MOGAC). This header
/// currently provides the MultiLevelSet data holder (with its factories
/// fromLabelGrid and fromSignedGrids and the exact per-label signed distance
/// extractSignedDistance), the second-order label-aware redistancer
/// MultiLevelSetRedistance, the LabelAwareStencil, the built-in speed functor
/// MultiObjectForces and the evolver LevelSetMultiObject.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/math/Math.h>
#include "Interpolation.h"// for BoxSampler
#include <openvdb/util/NullInterrupter.h>
#include "Morphology.h"// for dilateActiveValues, erodeActiveValues
#include "Prune.h"// for pruneLevelSet

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Holder for a pair of grids describing several disjoint objects:
/// an unsigned distance grid @c phi (distance to the nearest object
/// interface, in world units) and a grid of integer @c labels.
/// Label 0 denotes the background/passive region.
///
/// @details Invariant: both trees have identical active topology and phi is
/// finite and non-negative at every active voxel. phi has background value
/// <tt>halfWidth * voxelSize</tt> and the label background is 0.
template<typename FloatTreeT = FloatTree, typename LabelTreeT = Int32Tree>
class MultiLevelSet
{
public:
    using Ptr = SharedPtr<MultiLevelSet>;
    using FloatGridT = Grid<FloatTreeT>;
    using LabelGridT = Grid<LabelTreeT>;
    using LabelT = typename LabelTreeT::ValueType;

    /// @brief Pair up @a phi and @a labels.
    /// @throw ValueError if the transforms differ, the transform is not a
    /// uniform scale, or @a halfWidth is less than 2.
    MultiLevelSet(typename FloatGridT::Ptr phi, typename LabelGridT::Ptr labels, int halfWidth)
        : mPhi(phi), mLabels(labels), mHalfWidth(halfWidth)
    {
        if (!mPhi || !mLabels) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: null grid");
        }
        if (halfWidth < 2) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: halfWidth must be at least 2");
        }
        if (mPhi->transform() != mLabels->transform()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: phi and label grids must share a transform");
        }
        if (!mPhi->transform().hasUniformScale()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: transform must have uniform voxel size");
        }
        mPhi->setGridClass(GRID_UNKNOWN);
        mPhi->insertMeta("multi_level_set_role", StringMetadata("phi"));
        mLabels->insertMeta("multi_level_set_role", StringMetadata("labels"));
    }

    /// @brief Build a multi-object level set from a grid of labels.
    /// @details The label of a voxel is @a labels' value there (active or
    /// not); label 0 is background. Interface voxels (a voxel with a face
    /// neighbour of a different label) are searched among the active voxels
    /// of @a labels and their face neighbours and get phi = 0.5*dx (the voxel
    /// face midpoint), kept frozen while the rest of the band (interface
    /// voxels dilated by @a halfWidth over 26-neighbours) is redistanced
    /// (spec 5.2; no 5.1 reseed, whose gradient estimate needs a distance
    /// field rather than this staircase); voxels
    /// left with phi >= halfWidth*dx are deactivated. Every inactive voxel
    /// kept in the trees holds its label from @a labels (ruling R11).
    /// @throw ValueError if @a halfWidth < 2, the voxel size is not uniform
    /// or the label background is not 0.
    static Ptr fromLabelGrid(const LabelGridT& labels, int halfWidth = 3);

    /// @brief Build a multi-object level set from signed distance grids, one
    /// per non-zero label.
    /// @details psi_l is the input grid of label l (inactive values read
    /// with their sign, so deep interiors are negative); the background
    /// label 0 has the implicit psi_0 = -min_l psi_l (ruling R2). At each
    /// voxel the label is argmin psi (ties to the smaller label) and
    /// phi = (v_b - v_a)/2 for the two smallest values v_a <= v_b, so
    /// phi = |psi_l| near an isolated object. Candidates are the union of the
    /// input active topologies and their face neighbours; the band is the
    /// interface voxels dilated by @a halfWidth (26-neighbours), then
    /// reinitialized (spec 5) and trimmed to phi < halfWidth*dx. Inactive
    /// tiles and voxels kept in the label tree hold the deep-interior label
    /// (ruling R11). Inactive tiles are labelled by the argmin at the tile
    /// origin, which is exact for disjoint inputs; where inputs overlap
    /// inside a tile, only voxels in leaves (of the band, its face shell or
    /// an input) are labelled per voxel.
    /// @throw ValueError if the map is empty, a grid is null or not a
    /// GRID_LEVEL_SET, a key is 0, the transforms differ or are not a uniform
    /// scale, or @a halfWidth < 2.
    static Ptr fromSignedGrids(
        const std::map<int32_t, typename FloatGridT::ConstPtr>& grids, int halfWidth = 3);

    typename FloatGridT::Ptr phi() { return mPhi; }
    typename FloatGridT::ConstPtr phi() const { return mPhi; }
    typename LabelGridT::Ptr labels() { return mLabels; }
    typename LabelGridT::ConstPtr labels() const { return mLabels; }

    int halfWidth() const { return mHalfWidth; }
    double voxelSize() const { return mPhi->voxelSize()[0]; }

    /// @brief Raw signed distance of @a label at @a ijk: -phi inside the
    /// label, +phi elsewhere. Accurate only where the label's boundary is
    /// the nearest interface.
    float signedDistance(const Coord& ijk, int32_t label) const
    {
        const float d = mPhi->tree().getValue(ijk);
        return (mLabels->tree().getValue(ijk) == label) ? -d : d;
    }

    /// @brief Signed distance to the boundary of @a label (spec 5.3), exact
    /// also near junctions where the raw signedDistance() underestimates.
    /// @details A voxel is inside if its label (active or not, ruling R7b)
    /// equals @a label. Seeds are the active phi voxels with a face neighbour
    /// on the other side of that region. They keep phi (no 5.1 reseed),
    /// except outside seeds next to a third label, which are rebuilt from
    /// label-side values within proven bounds (lsmo_internal::
    /// rebuildJunctionSeeds, rulings R17, R19; first order at junction lines);
    /// the outer layers solve spec 5.2 on |psi|, and the sign is applied last.
    /// The result is active exactly at the voxels with |psi| < halfWidth*dx
    /// (the seeds dilated by @a halfWidth; for @a halfWidth up to this
    /// object's halfWidth() the search is limited to the active band, which
    /// already contains every such voxel). Inactive values are -/+
    /// halfWidth*dx by region.
    /// @param halfWidth band half-width in voxels; -1 uses halfWidth().
    /// @return a GRID_LEVEL_SET grid with background halfWidth*dx and this
    /// object's transform; it has no active voxels if @a label has no
    /// boundary in the band (e.g. it does not occur).
    /// @throw ValueError if @a halfWidth is not -1 and less than 2.
    typename FloatGridT::Ptr extractSignedDistance(int32_t label, int halfWidth = -1) const;

    /// @brief Sorted distinct non-zero labels over all active voxels.
    std::vector<int32_t> labelSet() const
    {
        std::set<int32_t> s;
        for (auto it = mLabels->tree().cbeginValueOn(); it; ++it) {
            s.insert(static_cast<int32_t>(*it));
        }
        s.erase(0);
        return std::vector<int32_t>(s.begin(), s.end());
    }

    /// @brief Check the class invariant.
    /// @param message if non-null, receives a description of the first failure.
    bool validate(std::string* message = nullptr) const
    {
        if (!mPhi->tree().hasSameTopology(mLabels->tree())) {
            if (message) *message = "phi and label trees have different active topology";
            return false;
        }
        for (auto it = mPhi->tree().cbeginValueOn(); it; ++it) {
            const float v = *it;
            if (!std::isfinite(v) || v < 0.0f) {
                if (message) {
                    std::ostringstream ss;
                    ss << "phi is negative or non-finite at " << it.getCoord();
                    *message = ss.str();
                }
                return false;
            }
        }
        return true;
    }

    /// @brief Dilate the active topology of both trees by @a iterations
    /// face-neighbour steps. New phi voxels get the background value; new
    /// label voxels copy the label of the active face neighbour with the
    /// smallest phi.
    void dilate(int iterations)
    {
        for (int n = 0; n < iterations; ++n) {
            FloatTreeT old(mPhi->tree(), mPhi->tree().background(), TopologyCopy());
            tools::dilateActiveValues(mPhi->tree(), 1, NN_FACE, IGNORE_TILES);
            tools::dilateActiveValues(mLabels->tree(), 1, NN_FACE, IGNORE_TILES);

            tree::ValueAccessor<const FloatTreeT> oldAcc(old);
            tree::ValueAccessor<FloatTreeT> phiAcc(mPhi->tree());
            tree::ValueAccessor<const FloatTreeT> phiRead(mPhi->tree());
            tree::ValueAccessor<LabelTreeT> labAcc(mLabels->tree());
            const float bg = mPhi->tree().background();
            static const Coord offs[6] = {
                Coord(-1,0,0), Coord(1,0,0), Coord(0,-1,0),
                Coord(0,1,0), Coord(0,0,-1), Coord(0,0,1)};
            for (auto it = mPhi->tree().beginValueOn(); it; ++it) {
                const Coord ijk = it.getCoord();
                if (oldAcc.isValueOn(ijk)) continue;
                float best = std::numeric_limits<float>::max();
                LabelT label = mLabels->tree().background();
                for (const Coord& o : offs) {
                    const Coord nb = ijk + o;
                    if (!oldAcc.isValueOn(nb)) continue;
                    const float d = phiRead.getValue(nb);
                    if (d < best) { best = d; label = labAcc.getValue(nb); }
                }
                phiAcc.setValueOn(ijk, bg);
                labAcc.setValueOn(ijk, label);
            }
        }
    }

    /// @brief Erode the active topology of both trees by @a iterations
    /// face-neighbour steps, then call pruneInactive().
    void erode(int iterations)
    {
        tools::erodeActiveValues(mPhi->tree(), iterations, NN_FACE, IGNORE_TILES);
        tools::erodeActiveValues(mLabels->tree(), iterations, NN_FACE, IGNORE_TILES);
        pruneInactive();
    }

    /// @brief Clean up inactive values. Every inactive phi voxel is reset to
    /// the phi background. Inactive label voxels keep their value (it records
    /// the nearby interior label). A leaf with no active voxels is replaced by
    /// an inactive tile in both trees only if its label leaf is uniform;
    /// all other leaves stay, so both trees keep the same leaf layout
    /// (the label tree is never pruned where voxels are active).
    void pruneInactive()
    {
        const float bg = mPhi->tree().background();
        std::vector<Coord> collapse;
        for (auto leaf = mPhi->tree().beginLeaf(); leaf; ++leaf) {
            for (auto it = leaf->beginValueOff(); it; ++it) it.setValue(bg);
            if (!leaf->isEmpty()) continue;
            const auto* lab = mLabels->tree().probeConstLeaf(leaf->origin());
            LabelT value; bool state;
            if (!lab || (lab->isConstant(value, state) && !state)) {
                collapse.push_back(leaf->origin());
            }
        }
        for (const Coord& o : collapse) {
            const auto* lab = mLabels->tree().probeConstLeaf(o);
            LabelT value = mLabels->tree().background(); bool state;
            if (lab) lab->isConstant(value, state);
            mPhi->tree().addTile(1, o, bg, false);
            mLabels->tree().addTile(1, o, value, false);
        }
    }

private:
    using BoolTreeT = typename FloatTreeT::template ValueConverter<bool>::Type;

    /// Shared factory back end; see the definition.
    template<typename SamplerT>
    static Ptr build(const math::Transform& xform, typename LabelTreeT::Ptr labels,
        const BoolTreeT& candidates, const SamplerT& sampler, int halfWidth, bool reseed);

    typename FloatGridT::Ptr mPhi;
    typename LabelGridT::Ptr mLabels;
    int mHalfWidth;
};


namespace lsmo_internal {

/// Face-neighbour offsets, ordered (-x, +x, -y, +y, -z, +z).
inline const Coord* faceOffsets()
{
    static const Coord offs[6] = {
        Coord(-1,0,0), Coord(1,0,0), Coord(0,-1,0),
        Coord(0,1,0), Coord(0,0,-1), Coord(0,0,1)};
    return offs;
}

/// @brief Interface seeding (spec 5.1) over the active voxels of @a out.
/// @details A voxel p is a seed if differs(p, q) holds for some face neighbour
/// q (active or not). Its value is the Russo-Smereka subcell fix (Russo &
/// Smereka 2000; analysed by du Chene, Min & Gibou 2008) applied to the
/// signed field psi of p's side: psi = -phi at p and at neighbours q with
/// !differs(p, q), +phi at neighbours with differs(p, q). Per axis
/// Delta_i = max(|psi(+1) - psi(-1)|/2, |psi(+1) - psi(0)|, |psi(0) - psi(-1)|,
/// 1e-6*dx) / dx and the seed is |psi_p| / sqrt(sum_i Delta_i^2); terms that
/// involve an inactive neighbour are left out (only band edges of finite
/// domains have them). A seed with no active face neighbour q with
/// differs(p, q) keeps its value (Delta could collapse to epsilon), and seed
/// values are clamped to the snapshot background.
/// All phi values are read from @a snapshot, so the result does not depend
/// on evaluation order. Seed
/// values are written to @a out and flagged as active voxels of @a seedMask;
/// nothing else in @a out changes. Active tiles of @a out are ignored.
/// @param differs  copied once per task; called as differs(p, q).
/// @param writeValues if false, only @a seedMask is built and @a out is
/// left untouched (used to freeze the current interface values).
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp>
void seedInterface(const FloatTreeT& snapshot, const DiffersOp& differs,
    FloatTreeT& out, BoolTreeT& seedMask, double dx, int grainSize, bool writeValues = true)
{
    using LeafT = typename FloatTreeT::LeafNodeType;
    seedMask.clear();
    seedMask.topologyUnion(out);
    seedMask.voxelizeActiveTiles();
    for (auto leaf = seedMask.beginLeaf(); leaf; ++leaf) leaf->setValuesOff();

    tree::LeafManager<FloatTreeT> leafs(out);
    const float bg = snapshot.background();
    const double eps = 1e-6 * dx;
    const Coord* offs = faceOffsets();

    auto body = [&](const typename tree::LeafManager<FloatTreeT>::LeafRange& range) {
        DiffersOp op = differs; // per-task copy (may own value accessors)
        tree::ValueAccessor<const FloatTreeT> acc(snapshot);
        for (auto leafIt = range.begin(); leafIt; ++leafIt) {
            LeafT& leaf = *leafIt;
            auto* maskLeaf = seedMask.probeLeaf(leaf.origin());
            for (auto it = leaf.beginValueOn(); it; ++it) {
                const Coord p = it.getCoord();
                // psi at the six face neighbours, signed relative to p's side;
                // inactive neighbours are left out of the gradient.
                double psi[6];
                bool on[6], seed = false, otherOn = false;
                for (int f = 0; f < 6; ++f) {
                    const Coord q = p + offs[f];
                    const bool other = op(p, q);
                    seed |= other;
                    if (!writeValues) { if (seed) break; continue; }
                    float phiQ = bg;
                    on[f] = acc.probeValue(q, phiQ);
                    otherOn |= on[f] && other;
                    psi[f] = other ? double(phiQ) : -double(phiQ);
                }
                if (!seed) continue;
                if (writeValues && otherOn) {
                    const double psiP = -double(acc.getValue(p));
                    double sum2 = 0.0;
                    for (int axis = 0; axis < 3; ++axis) {
                        const int fm = 2 * axis, fp = 2 * axis + 1;
                        double d = eps;
                        if (on[fm] && on[fp]) d = std::max(d, std::abs(psi[fp] - psi[fm]) * 0.5);
                        if (on[fp]) d = std::max(d, std::abs(psi[fp] - psiP));
                        if (on[fm]) d = std::max(d, std::abs(psiP - psi[fm]));
                        d /= dx;
                        sum2 += d * d;
                    }
                    it.setValue(float(std::min(std::abs(psiP) / std::sqrt(sum2), double(bg))));
                }
                if (maskLeaf) maskLeaf->setValueOn(it.pos(), true);
            }
        }
    };
    if (grainSize > 0) tbb::parallel_for(leafs.leafRange(size_t(grainSize)), body);
    else body(leafs.leafRange());
}

/// Seeding predicate for MultiLevelSetRedistance: labels of p and q differ.
template<typename LabelTreeT>
struct LabelDiffers
{
    explicit LabelDiffers(const LabelTreeT& t): acc(t) {}
    LabelDiffers(const LabelDiffers& o): acc(o.acc.tree()) {}
    bool operator()(const Coord& p, const Coord& q) { return acc.getValue(p) != acc.getValue(q); }
    tree::ValueAccessor<const LabelTreeT> acc;
};

/// Seeding predicate for the IsInsideOp form: region membership differs.
template<typename IsInsideOp>
struct InsideDiffers
{
    explicit InsideDiffers(const IsInsideOp& op): inside(op) {}
    bool operator()(const Coord& p, const Coord& q) { return inside(p) != inside(q); }
    IsInsideOp inside;
};


/// Marker for voxels not yet reached by the outer-layer solver.
constexpr float kUnknownDistance = std::numeric_limits<float>::max();

/// Solve sum_i c_i (phi - abar_i)^2 = dx^2 over the sorted upwind set,
/// dropping terms with phi <= abar_i.
inline double solveGodunov(double* abar, double* coef, int n, double dx)
{
    for (int i = 1; i < n; ++i) { // insertion sort by abar
        for (int j = i; j > 0 && abar[j] < abar[j - 1]; --j) {
            std::swap(abar[j], abar[j - 1]);
            std::swap(coef[j], coef[j - 1]);
        }
    }
    const double h2 = dx * dx;
    double result = abar[0] + dx / std::sqrt(coef[0]);
    double A = coef[0], B = coef[0] * abar[0], C = coef[0] * abar[0] * abar[0];
    for (int k = 1; k < n; ++k) {
        if (result <= abar[k]) break;
        A += coef[k];
        B += coef[k] * abar[k];
        C += coef[k] * abar[k] * abar[k];
        // A phi^2 - 2 B phi + (C - h^2) = 0
        const double disc = B * B - A * (C - h2);
        if (disc < 0.0) break;
        result = (B + std::sqrt(disc)) / A;
    }
    return result;
}

/// @brief Outer-layer solver (spec 5.2) on the unsigned distance @a phi.
/// @details Active voxels of @a phi that are not active in @a seeds are reset
/// and solve |grad phi| = 1 by parallel Jacobi iteration with the seeds as
/// Dirichlet data: second-order one-sided upwind differences (Chopp 2001)
/// where two same-side upwind values are available (the outer one, q2, with
/// !differs(p, q2) and strictly smaller than the inner one) and first order
/// otherwise. Inactive neighbours are not used. Iteration stops when
/// max|delta phi| < @a tolerance, after @a maxIter iterations, or on
/// interrupt; voxels left unreached get the background of @a phi.
/// @param differs copied once per task; called as differs(p, q).
/// @param grainSize TBB grain size in leaves; zero or negative runs serially.
/// @return the number of Jacobi iterations performed.
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp, typename InterruptT>
int solveOuterLayers(FloatTreeT& phi, const BoolTreeT& seeds, const DiffersOp& differs,
    double dx, float tolerance, int maxIter, int grainSize, InterruptT* interrupt)
{
    using LeafManagerT = tree::LeafManager<FloatTreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;
    const float bg = phi.background();
    const bool serial = grainSize <= 0;
    const Coord* offs = faceOffsets();

    auto forEachLeaf = [&](LeafManagerT& leafs, const auto& op) {
        auto body = [&](const LeafRangeT& range) {
            for (auto leafIt = range.begin(); leafIt; ++leafIt) op(leafIt);
        };
        if (grainSize > 0) tbb::parallel_for(leafs.leafRange(size_t(grainSize)), body);
        else body(leafs.leafRange());
    };

    // Reset non-seed voxels; Jacobi with one aux buffer.
    LeafManagerT leafs(phi, 1, serial);
    forEachLeaf(leafs, [&](typename LeafRangeT::Iterator& leafIt) {
        const auto* mask = seeds.probeConstLeaf(leafIt->origin());
        for (auto it = leafIt->beginValueOn(); it; ++it) {
            if (!mask || !mask->isValueOn(it.pos())) it.setValue(kUnknownDistance);
        }
    });
    leafs.syncAllBuffers(serial);

    // One Jacobi iteration: read main buffers, write aux buffer 1; returns
    // max|delta phi| over updated voxels.
    auto sweep = [&]() -> float {
        auto body = [&](const LeafRangeT& range, float maxDelta) {
            DiffersOp op = differs; // per-task copy (may own value accessors)
            tree::ValueAccessor<const FloatTreeT> acc(phi);
            for (auto leafIt = range.begin(); leafIt; ++leafIt) {
                auto& aux = leafIt.buffer(1);
                const auto* mask = seeds.probeConstLeaf(leafIt->origin());
                for (auto it = leafIt->cbeginValueOn(); it; ++it) {
                    if (mask && mask->isValueOn(it.pos())) continue;
                    const Coord p = it.getCoord();
                    const float old = *it;
                    double abar[3], coef[3];
                    int n = 0;
                    for (int axis = 0; axis < 3; ++axis) {
                        float a = kUnknownDistance, a2 = kUnknownDistance;
                        for (int side = 0; side < 2; ++side) {
                            float v;
                            const Coord q = p + offs[2 * axis + side];
                            if (!acc.probeValue(q, v) || !(v < a)) continue;
                            a = v;
                            // a' must lie on p's side of the interface
                            const Coord q2 = q + offs[2 * axis + side];
                            if (!acc.probeValue(q2, a2) || op(p, q2)) a2 = kUnknownDistance;
                        }
                        if (a == kUnknownDistance) continue;
                        if (a2 < a) { // second order (Chopp 2001)
                            abar[n] = (4.0 * a - a2) / 3.0;
                            coef[n] = 9.0 / 4.0;
                        } else {
                            abar[n] = a;
                            coef[n] = 1.0;
                        }
                        ++n;
                    }
                    if (n == 0) { aux.setValue(it.pos(), old); continue; }
                    const float upd = float(solveGodunov(abar, coef, n, dx));
                    const float val = std::min(old, upd);
                    aux.setValue(it.pos(), val);
                    maxDelta = std::max(maxDelta, old - val);
                }
            }
            return maxDelta;
        };
        if (grainSize > 0) {
            return tbb::parallel_reduce(leafs.leafRange(size_t(grainSize)), 0.0f, body,
                [](float a, float b) { return std::max(a, b); });
        }
        return body(leafs.leafRange(), 0.0f);
    };

    int iter = 0;
    while (iter < maxIter) {
        if (util::wasInterrupted(interrupt)) break;
        ++iter;
        const float maxDelta = sweep();
        leafs.swapLeafBuffer(1, serial);
        if (maxDelta < tolerance) break;
    }

    // Unreached voxels (no seed nearby, or interrupted) get the background.
    forEachLeaf(leafs, [&](typename LeafRangeT::Iterator& leafIt) {
        for (auto it = leafIt->beginValueOn(); it; ++it) {
            if (*it == kUnknownDistance) it.setValue(bg);
        }
    });
    return iter;
}

} // namespace lsmo_internal


/// @brief Second-order label-aware redistancing of a MultiLevelSet (spec 5).
///
/// @details Restores phi = distance to the voxel's own label boundary over
/// the active band while keeping the interface fixed:
/// -# Interface voxels (an active voxel with a face neighbour, active or
///    not, of a different label) are frozen. redistance() keeps their
///    current values, so repeated calls never move a crossing (use it every
///    evolution step). reinitialize() first reseeds them with the spec 5.1
///    Russo-Smereka subcell distance, computed from a snapshot of phi; the
///    reseed is not a fixed point of itself on curved interfaces, so
///    reinitialize() is meant for one-shot use (factories, per-label
///    reconstruction).
/// -# All other active voxels solve |grad phi| = 1 by parallel Jacobi
///    iteration, using second-order one-sided upwind differences (Chopp 2001)
///    where two same-side upwind values are available (the outer one with
///    p's label and strictly smaller than the inner one) and first order
///    otherwise. Inactive neighbours are not used. Iteration stops when
///    max|delta phi| falls below the tolerance or after
///    ceil(halfWidth*sqrt(3)) + 2 iterations.
///
/// Only active phi voxels are written; topology, labels and inactive values
/// are untouched. Voxels with no path to a seed get the phi background.
/// Outer values are not clamped and may exceed the background
/// (halfWidth*dx); deactivating such voxels is left to band maintenance.
///
/// @pre The label tree holds the correct (deep-interior) label at the
/// inactive face neighbours of the band (ruling R7b). A band-edge voxel
/// whose inactive neighbour reads the label background instead becomes a
/// spurious interface voxel.
///
/// @note FastSweeping.h was evaluated for reuse (spec 5.2) and rejected: its
/// update is first-order Godunov only, it sweeps in place (Gauss-Seidel), and
/// its boundary voxels are found from sign changes of the input field (or a
/// mask) through private initialisers, so an all-positive field with
/// externally frozen seeds and a second-order stencil cannot be expressed
/// without modifying that header. A dedicated band Jacobi kernel is used.
template<typename MultiLevelSetT, typename InterruptT = util::NullInterrupter>
class MultiLevelSetRedistance
{
public:
    using FloatTreeT = typename MultiLevelSetT::FloatGridT::TreeType;
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;
    using BoolTreeT = typename FloatTreeT::template ValueConverter<bool>::Type;

    explicit MultiLevelSetRedistance(MultiLevelSetT& mls, InterruptT* interrupt = nullptr)
        : mMls(mls), mInterrupter(interrupt), mTolerance(float(1e-4 * mls.voxelSize()))
    {
    }

    /// @brief Convergence tolerance on max|delta phi| in world units
    /// (default 1e-4*dx).
    void setTolerance(float tolWorld) { mTolerance = tolWorld; }
    /// @brief TBB grain size in leaves; zero or negative runs serially.
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }

    /// @brief Redistance phi in place, keeping the interface-voxel values
    /// frozen (idempotent: crossings never move).
    /// @return the number of Jacobi iterations performed.
    int redistance() { return this->run(/*reseed=*/false); }

    /// @brief Reseed the interface voxels with the spec 5.1 subcell distance,
    /// then redistance phi in place.
    /// @return the number of Jacobi iterations performed.
    int reinitialize() { return this->run(/*reseed=*/true); }

    /// @brief Interface seeding only (spec 5.1), for per-label reconstruction.
    /// @details An active voxel p of @a out is a seed if some face neighbour
    /// q has isInside(p) != isInside(q). Its value is the Russo-Smereka
    /// subcell distance |psi_p| / |grad psi| (see lsmo_internal::seedInterface)
    /// of psi = -phi where isInside, +phi elsewhere, with phi from
    /// @a phiSnapshot. Seeds are written to @a out and marked active in
    /// @a seedMask (which is overwritten); other voxels of @a out are unchanged.
    /// @param voxelSize world-space voxel size dx (trees carry no transform,
    /// and the gradient estimate needs it).
    /// @param grainSize TBB grain size in leaves; zero or negative runs serially.
    /// @a isInside is copied once per task, so it may own value accessors.
    template<typename IsInsideOp>
    static void seedInterface(const FloatTreeT& phiSnapshot, IsInsideOp isInside,
        FloatTreeT& out, BoolTreeT& seedMask, double voxelSize, int grainSize = 1)
    {
        lsmo_internal::seedInterface(phiSnapshot,
            lsmo_internal::InsideDiffers<IsInsideOp>(isInside), out, seedMask,
            voxelSize, grainSize);
    }

private:
    int run(bool reseed)
    {
        FloatTreeT& phi = mMls.phi()->tree();
        const double dx = mMls.voxelSize();
        const int maxIter = int(std::ceil(mMls.halfWidth() * std::sqrt(3.0))) + 2;

        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");

        // Interface voxels: frozen as they are, or reseeded (5.1) from a
        // read-only snapshot.
        BoolTreeT seeds(false);
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        if (reseed) {
            const FloatTreeT snapshot(phi);
            lsmo_internal::seedInterface(snapshot, differs, phi, seeds, dx, mGrainSize);
        } else {
            lsmo_internal::seedInterface(phi, differs, phi, seeds, dx, mGrainSize,
                /*writeValues=*/false);
        }

        // 5.2: outer layers
        const int iter = lsmo_internal::solveOuterLayers(phi, seeds, differs, dx,
            mTolerance, maxIter, mGrainSize, mInterrupter);

        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    MultiLevelSetT& mMls;
    InterruptT* mInterrupter;
    float mTolerance;
    int mGrainSize = 1;
};


////////////////////////////////////////


namespace lsmo_internal {

/// @brief Predicate for MultiLevelSet::extractSignedDistance: membership in
/// the region {L == label} differs between p and q (labels read with
/// getValue, so inactive labels count). With @a requireActive, p must also
/// be an active voxel of @a phi. Copied once per task.
template<typename FloatTreeT, typename LabelTreeT>
struct LabelRegionDiffers
{
    using LabelT = typename LabelTreeT::ValueType;
    LabelRegionDiffers(const LabelTreeT& labels, const FloatTreeT& phi, LabelT l, bool active)
        : lab(labels), phiAcc(phi), label(l), requireActive(active) {}
    LabelRegionDiffers(const LabelRegionDiffers& o)
        : lab(o.lab.tree()), phiAcc(o.phiAcc.tree()), label(o.label)
        , requireActive(o.requireActive) {}

    bool operator()(const Coord& p, const Coord& q)
    {
        if (requireActive && !phiAcc.isValueOn(p)) return false;
        return (lab.getValue(p) == label) != (lab.getValue(q) == label);
    }

    tree::ValueAccessor<const LabelTreeT> lab;
    tree::ValueAccessor<const FloatTreeT> phiAcc;
    LabelT label;
    bool requireActive;
};

/// @brief Per-label reconstruction near junctions (spec 5.3, rulings R17, R19).
/// @details Let psi be the signed distance to the region {L == @a label}
/// and q a seed of @a out (active in @a seeds) outside the region. Its value
/// phi_q is the distance to its own label's boundary, which next to a third
/// label may be a boundary with that label rather than the region's. Two
/// bounds hold whenever region voxels store their exact distance to the
/// region's boundary (which is that label's own boundary):
/// - lower: phi_q <= psi(q), since the region's boundary seen from q is part
///   of the boundary of L(q)'s region;
/// - upper: psi(q) <= |q - p| - phi_p for every region voxel p, since the
///   ball of radius phi_p about p lies inside the region.
/// Every outside seed with a face neighbour (active or not) of a third label
/// (neither @a label nor L(q)) is rebuilt by a local Godunov extension of
/// psi: per axis the smallest known psi among the two face neighbours
/// (-phi at region seeds, +value at other outside seeds and at rebuilt seeds
/// once solved), then sum_i (psi - a_i)^2 = dx^2, clamped to
/// [phi_q, min(B, background)] with B the upper bound over the active region
/// voxels p within the 5x5x5 neighbourhood of q. Jacobi sweeps over just
/// these seeds until no value changes by more than 1e-4*dx (at most 8). A
/// seed with no known neighbour keeps phi_q. psi is not smooth at junction
/// lines, so the rebuilt values are only first-order accurate there; seeds
/// whose phi_q is already exact stay within the interval but may move.
/// Runs serially (junction seeds are few).
template<typename FloatTreeT, typename BoolTreeT, typename LabelTreeT>
void rebuildJunctionSeeds(FloatTreeT& out, const BoolTreeT& seeds, const LabelTreeT& labels,
    typename LabelTreeT::ValueType label, double dx)
{
    const Coord* offs = faceOffsets();
    tree::ValueAccessor<const LabelTreeT> lab(labels);
    tree::ValueAccessor<FloatTreeT> acc(out);
    tree::ValueAccessor<const BoolTreeT> seedAcc(seeds);

    const float bg = out.background();
    std::vector<Coord> todo;
    std::vector<float> keep, upper;
    for (auto it = seeds.cbeginValueOn(); it; ++it) {
        const Coord q = it.getCoord();
        const auto lq = lab.getValue(q);
        if (lq == label) continue;
        bool third = false;
        for (int f = 0; f < 6 && !third; ++f) {
            const auto ln = lab.getValue(q + offs[f]);
            third = ln != label && ln != lq;
        }
        if (!third) continue;
        // upper bound from the balls inside the region (see above)
        const float phiQ = acc.getValue(q);
        double ball = double(bg);
        for (int k = -2; k <= 2; ++k) {
            for (int j = -2; j <= 2; ++j) {
                for (int i = -2; i <= 2; ++i) {
                    const Coord p = q + Coord(i, j, k);
                    float v;
                    if (lab.getValue(p) != label || !acc.probeValue(p, v)) continue;
                    ball = std::min(ball, dx * std::sqrt(double(i * i + j * j + k * k)) - double(v));
                }
            }
        }
        todo.push_back(q);
        keep.push_back(phiQ);
        upper.push_back(float(std::max(ball, double(phiQ))));
    }
    if (todo.empty()) return;
    for (const Coord& q : todo) acc.setValue(q, kUnknownDistance);

    const float tol = float(1e-4 * dx);
    std::vector<float> next(todo.size());
    for (int sweep = 0; sweep < 8; ++sweep) {
        float maxDelta = 0.0f;
        for (size_t n = 0; n < todo.size(); ++n) {
            const Coord& q = todo[n];
            double abar[3], coef[3] = {1.0, 1.0, 1.0};
            int m = 0;
            for (int axis = 0; axis < 3; ++axis) {
                double a = std::numeric_limits<double>::max();
                for (int side = 0; side < 2; ++side) {
                    const Coord nb = q + offs[2 * axis + side];
                    if (!seedAcc.isValueOn(nb)) continue;
                    const float v = acc.getValue(nb);
                    if (v == kUnknownDistance) continue;
                    a = std::min(a, lab.getValue(nb) == label ? -double(v) : double(v));
                }
                if (a != std::numeric_limits<double>::max()) abar[m++] = a;
            }
            const float old = acc.getValue(q);
            next[n] = m == 0 ? old
                : float(std::min(std::max(solveGodunov(abar, coef, m, dx), double(keep[n])),
                    double(upper[n])));
            if (old == kUnknownDistance) {
                if (next[n] != kUnknownDistance) maxDelta = std::numeric_limits<float>::max();
            } else {
                maxDelta = std::max(maxDelta, std::abs(next[n] - old));
            }
        }
        for (size_t n = 0; n < todo.size(); ++n) acc.setValue(todo[n], next[n]);
        if (maxDelta <= tol) break;
    }
    for (size_t n = 0; n < todo.size(); ++n) {
        if (acc.getValue(todo[n]) == kUnknownDistance) acc.setValue(todo[n], keep[n]);
    }
}

/// @brief Factory sampler for MultiLevelSet::fromLabelGrid: the label is the
/// input value; phi is 0.5*dx at voxels with a face neighbour of a different
/// label and @a bg elsewhere. Copied once per task.
template<typename LabelTreeT>
struct LabelGridSampler
{
    LabelGridSampler(const LabelTreeT& t, double dx, float bg)
        : acc(t), half(float(0.5 * dx)), background(bg) {}
    LabelGridSampler(const LabelGridSampler& o)
        : acc(o.acc.tree()), half(o.half), background(o.background) {}

    int32_t label(const Coord& p) { return static_cast<int32_t>(acc.getValue(p)); }
    float phi(const Coord& p)
    {
        const auto l = acc.getValue(p);
        const Coord* offs = faceOffsets();
        for (int f = 0; f < 6; ++f) {
            if (acc.getValue(p + offs[f]) != l) return half;
        }
        return background;
    }

    tree::ValueAccessor<const LabelTreeT> acc;
    float half, background;
};

/// @brief Factory sampler for MultiLevelSet::fromSignedGrids. Candidates are
/// (psi_l, l) for every input and (psi_0 = -min_l psi_l, 0) for the
/// background (ruling R2); the label is the smallest candidate (ties to the
/// smaller label) and phi = (v_b - v_a)/2 for the two smallest v_a <= v_b.
/// Copied once per task.
template<typename FloatTreeT>
struct SignedGridsSampler
{
    using Input = std::pair<int32_t, const FloatTreeT*>;

    explicit SignedGridsSampler(const std::vector<Input>& in): inputs(in) { this->init(); }
    SignedGridsSampler(const SignedGridsSampler& o): inputs(o.inputs) { this->init(); }

    void eval(const Coord& p, int32_t& label, double& phi)
    {
        const double inf = std::numeric_limits<double>::infinity();
        double va = inf, vb = inf, minPsi = inf;
        int32_t la = 0, lb = 0;
        auto push = [&](double v, int32_t l) {
            if (v < va || (v == va && l < la)) {
                vb = va; lb = la; va = v; la = l;
            } else if (v < vb || (v == vb && l < lb)) {
                vb = v; lb = l;
            }
        };
        for (size_t i = 0; i < inputs.size(); ++i) {
            const double psi = accs[i].getValue(p);
            minPsi = std::min(minPsi, psi);
            push(psi, inputs[i].first);
        }
        push(-minPsi, 0);
        label = la;
        phi = 0.5 * (vb - va);
    }
    int32_t label(const Coord& p) { int32_t l; double d; this->eval(p, l, d); return l; }
    float phi(const Coord& p) { int32_t l; double d; this->eval(p, l, d); return float(d); }

    std::vector<Input> inputs;
    std::vector<tree::ValueAccessor<const FloatTreeT>> accs;

private:
    void init()
    {
        accs.reserve(inputs.size());
        for (const Input& in : inputs) accs.emplace_back(*in.second);
    }
};

} // namespace lsmo_internal


/// @details Factory back end. @a labels holds the inactive structure (tiles
/// and leaves) whose values must carry the deep-interior labels; @a sampler
/// provides label(p) and the initial phi(p) anywhere.
/// -# Interface voxels: voxels of @a candidates with a face neighbour of a
///    different sampled label.
/// -# Band: interface voxels dilated by @a halfWidth (26-neighbours); every
///    label leaf touched by the band or its face shell gets the sampled
///    label at all its voxels, active exactly on the band.
/// -# phi mirrors the label tree's structure; band voxels get the sampled
///    phi; then reinitialize() (spec 5.1 reseed + 5.2 solve) if @a reseed,
///    else redistance() (sampled interface values frozen + 5.2 solve).
/// -# Voxels with phi >= halfWidth*dx are deactivated (labels kept), then
///    pruneInactive().
template<typename FloatTreeT, typename LabelTreeT>
template<typename SamplerT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::build(const math::Transform& xform,
    typename LabelTreeT::Ptr labels, const BoolTreeT& candidates,
    const SamplerT& sampler, int halfWidth, bool reseed)
{
    const double dx = xform.voxelSize()[0];
    const float bg = float(halfWidth * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // 1. interface voxels
    BoolTreeT band(candidates);
    band.voxelizeActiveTiles();
    {
        tree::LeafManager<BoolTreeT> leafs(band);
        tbb::parallel_for(leafs.leafRange(1),
            [&](const typename tree::LeafManager<BoolTreeT>::LeafRange& range) {
                SamplerT s(sampler);
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    for (auto it = leaf->beginValueOn(); it; ++it) {
                        const Coord p = it.getCoord();
                        const int32_t l = s.label(p);
                        bool iface = false;
                        for (int f = 0; f < 6 && !iface; ++f) iface = s.label(p + offs[f]) != l;
                        if (!iface) leaf->setValueOff(it.pos());
                    }
                }
            });
    }

    // 2. band and label tree
    tools::dilateActiveValues(band, halfWidth, NN_FACE_EDGE_VERTEX, IGNORE_TILES);
    {
        BoolTreeT shell(band);
        tools::dilateActiveValues(shell, 1, NN_FACE, IGNORE_TILES);
        tree::ValueAccessor<LabelTreeT> acc(*labels);
        for (auto leaf = shell.cbeginLeaf(); leaf; ++leaf) acc.touchLeaf(leaf->origin());
    }
    {
        tree::LeafManager<LabelTreeT> leafs(*labels);
        tbb::parallel_for(leafs.leafRange(1),
            [&](const typename tree::LeafManager<LabelTreeT>::LeafRange& range) {
                SamplerT s(sampler);
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    const auto* bandLeaf = band.probeConstLeaf(leaf->origin());
                    for (Index i = 0; i < LabelTreeT::LeafNodeType::NUM_VALUES; ++i) {
                        leaf->setValueOnly(i, LabelT(s.label(leaf->offsetToGlobalCoord(i))));
                        leaf->setActiveState(i, bandLeaf && bandLeaf->isValueOn(i));
                    }
                }
            });
    }

    // 3. phi with the same structure, then reinitialize
    typename FloatTreeT::Ptr phiTree(new FloatTreeT(*labels, bg, TopologyCopy()));
    {
        tree::LeafManager<FloatTreeT> leafs(*phiTree);
        tbb::parallel_for(leafs.leafRange(1),
            [&](const typename tree::LeafManager<FloatTreeT>::LeafRange& range) {
                SamplerT s(sampler);
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    for (auto it = leaf->beginValueOn(); it; ++it) it.setValue(s.phi(it.getCoord()));
                }
            });
    }
    typename FloatGridT::Ptr phiGrid = FloatGridT::create(phiTree);
    phiGrid->setTransform(xform.copy());
    typename LabelGridT::Ptr labGrid = LabelGridT::create(labels);
    labGrid->setTransform(xform.copy());
    Ptr mls(new MultiLevelSet(phiGrid, labGrid, halfWidth));
    MultiLevelSetRedistance<MultiLevelSet> redist(*mls);
    if (reseed) redist.reinitialize(); else redist.redistance();

    // 4. trim to the band phi < halfWidth*dx
    {
        tree::LeafManager<FloatTreeT> leafs(*phiTree);
        tbb::parallel_for(leafs.leafRange(1),
            [&](const typename tree::LeafManager<FloatTreeT>::LeafRange& range) {
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    auto* labLeaf = labels->probeLeaf(leaf->origin());
                    for (auto it = leaf->beginValueOn(); it; ++it) {
                        if (*it < bg) continue;
                        leaf->setValueOff(it.pos(), bg);
                        labLeaf->setActiveState(it.pos(), false);
                    }
                }
            });
    }
    mls->pruneInactive();
    return mls;
}


template<typename FloatTreeT, typename LabelTreeT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::FloatGridT::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::extractSignedDistance(int32_t label, int halfWidth) const
{
    if (halfWidth == -1) halfWidth = mHalfWidth;
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::extractSignedDistance: halfWidth must be at least 2");
    }
    using RegionDiffers = lsmo_internal::LabelRegionDiffers<FloatTreeT, LabelTreeT>;
    using BoolLeafRangeT = typename tree::LeafManager<BoolTreeT>::LeafRange;
    using LeafRangeT = typename tree::LeafManager<FloatTreeT>::LeafRange;
    const FloatTreeT& phi = mPhi->tree();
    const LabelTreeT& labels = mLabels->tree();
    const double dx = this->voxelSize();
    const float bg = float(halfWidth * dx);
    const int grainSize = 1;
    const LabelT inLabel = LabelT(label);
    const Coord* offs = lsmo_internal::faceOffsets();
    // Seeds must be active in phi (the pair holds distances only there);
    // the side test of the outer-layer solver is pure region membership.
    const RegionDiffers seedOp(labels, phi, inLabel, /*requireActive=*/true);
    const RegionDiffers sideOp(labels, phi, inLabel, /*requireActive=*/false);

    // 1. label | not-label interface voxels of the band, dilated by halfWidth
    BoolTreeT band(phi, false, TopologyCopy());
    band.voxelizeActiveTiles();
    {
        tree::LeafManager<BoolTreeT> leafs(band);
        tbb::parallel_for(leafs.leafRange(grainSize), [&](const BoolLeafRangeT& range) {
            RegionDiffers op(seedOp);
            for (auto leaf = range.begin(); leaf; ++leaf) {
                for (auto it = leaf->beginValueOn(); it; ++it) {
                    const Coord p = it.getCoord();
                    bool seed = false;
                    for (int f = 0; f < 6 && !seed; ++f) seed = op(p, p + offs[f]);
                    if (!seed) leaf->setValueOff(it.pos());
                }
            }
        });
    }
    tools::dilateActiveValues(band, halfWidth, NN_FACE_EDGE_VERTEX, IGNORE_TILES);
    if (halfWidth <= mHalfWidth) band.topologyIntersection(phi);

    // 2. output tree over the band, initialised with phi
    typename FloatTreeT::Ptr out(new FloatTreeT(band, bg, TopologyCopy()));
    {
        tree::LeafManager<FloatTreeT> leafs(*out);
        tbb::parallel_for(leafs.leafRange(grainSize), [&](const LeafRangeT& range) {
            tree::ValueAccessor<const FloatTreeT> acc(phi);
            for (auto leaf = range.begin(); leaf; ++leaf) {
                for (auto it = leaf->beginValueOn(); it; ++it) it.setValue(acc.getValue(it.getCoord()));
            }
        });
    }

    // 3. seeds: label | not-label crossings, keeping the pair's phi (ruling
    // R17: no subcell reseed, which is not a fixed point of reinitialized
    // phi). Not-label seeds next to a third label may store the distance to
    // the wrong boundary and are rebuilt from label-side data.
    BoolTreeT seeds(false);
    lsmo_internal::seedInterface(phi, seedOp, *out, seeds, dx, grainSize,
        /*writeValues=*/false);
    lsmo_internal::rebuildJunctionSeeds(*out, seeds, labels, inLabel, dx);
    const int maxIter = int(std::ceil(halfWidth * std::sqrt(3.0))) + 2;
    lsmo_internal::solveOuterLayers(*out, seeds, sideOp, dx, float(1e-4 * dx), maxIter,
        grainSize, static_cast<util::NullInterrupter*>(nullptr));

    // 4. sign by region, trim to |psi| < halfWidth*dx, inactive values -/+bg
    {
        tree::LeafManager<FloatTreeT> leafs(*out);
        tbb::parallel_for(leafs.leafRange(grainSize), [&](const LeafRangeT& range) {
            tree::ValueAccessor<const LabelTreeT> lab(labels);
            for (auto leaf = range.begin(); leaf; ++leaf) {
                for (Index i = 0; i < FloatTreeT::LeafNodeType::NUM_VALUES; ++i) {
                    const bool inside = lab.getValue(leaf->offsetToGlobalCoord(i)) == inLabel;
                    const float v = leaf->getValue(i);
                    if (leaf->isValueOn(i) && v < bg) {
                        leaf->setValueOnly(i, inside ? -v : v);
                    } else {
                        leaf->setValueOff(i, inside ? -bg : bg);
                    }
                }
            }
        });
    }
    {
        tree::ValueAccessor<const LabelTreeT> lab(labels);
        auto it = out->beginValueOff();
        it.setMaxDepth(FloatTreeT::ValueOffIter::LEAF_DEPTH - 1); // tiles only
        for (; it; ++it) it.setValue(lab.getValue(it.getCoord()) == inLabel ? -bg : bg);
    }
    tools::pruneLevelSet(*out, /*threaded=*/true, grainSize);

    typename FloatGridT::Ptr grid = FloatGridT::create(out);
    grid->setTransform(mPhi->transform().copy());
    grid->setGridClass(GRID_LEVEL_SET);
    return grid;
}


template<typename FloatTreeT, typename LabelTreeT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::fromLabelGrid(const LabelGridT& labels, int halfWidth)
{
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: halfWidth must be at least 2");
    }
    if (!labels.transform().hasUniformScale()) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: transform must have uniform voxel size");
    }
    if (labels.background() != LabelT(0)) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: label background must be 0");
    }
    const double dx = labels.transform().voxelSize()[0];

    // Keep the input's structure and values (deep-interior labels), inactive.
    typename LabelTreeT::Ptr lab(new LabelTreeT(labels.tree()));
    for (auto it = lab->beginValueOn(); it; ++it) it.setValueOff();

    BoolTreeT candidates(labels.tree(), false, TopologyCopy());
    candidates.voxelizeActiveTiles();
    tools::dilateActiveValues(candidates, 1, NN_FACE, IGNORE_TILES);

    return build(labels.transform(), lab, candidates,
        lsmo_internal::LabelGridSampler<LabelTreeT>(labels.tree(), dx, float(halfWidth * dx)),
        halfWidth, /*reseed=*/false);
}


template<typename FloatTreeT, typename LabelTreeT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::fromSignedGrids(
    const std::map<int32_t, typename FloatGridT::ConstPtr>& grids, int halfWidth)
{
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: halfWidth must be at least 2");
    }
    if (grids.empty()) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: no input grids");
    }
    using SamplerT = lsmo_internal::SignedGridsSampler<FloatTreeT>;
    const math::Transform* xform = nullptr;
    std::vector<typename SamplerT::Input> inputs;
    for (const auto& kv : grids) {
        if (!kv.second) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: null input grid");
        }
        if (kv.first == 0) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: label 0 is reserved for the background");
        }
        if (kv.second->getGridClass() != GRID_LEVEL_SET) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: input grids must be level sets");
        }
        if (!xform) {
            xform = &kv.second->transform();
        } else if (*xform != kv.second->transform()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: input grids must share a transform");
        }
        inputs.emplace_back(kv.first, &kv.second->tree());
    }
    if (!xform->hasUniformScale()) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: transform must have uniform voxel size");
    }
    SamplerT sampler(inputs);

    // Deep interiors: every negative inactive tile of an input becomes a
    // tile of its argmin label (largest tiles first, so smaller ones refine
    // them); every input leaf is touched so its voxels get sampled labels.
    typename LabelTreeT::Ptr lab(new LabelTreeT(LabelT(0)));
    std::vector<std::pair<Index, Coord>> tiles;
    for (const auto& in : inputs) {
        for (auto it = in.second->cbeginValueOff(); it; ++it) {
            if (it.getLevel() == 0 || !(*it < 0)) continue;
            tiles.emplace_back(it.getLevel(), it.getCoord());
        }
    }
    std::stable_sort(tiles.begin(), tiles.end(),
        [](const std::pair<Index, Coord>& a, const std::pair<Index, Coord>& b) {
            return a.first > b.first;
        });
    for (const auto& t : tiles) {
        const int32_t l = sampler.label(t.second);
        if (l != 0) lab->addTile(t.first, t.second, LabelT(l), false);
    }
    {
        tree::ValueAccessor<LabelTreeT> acc(*lab);
        for (const auto& in : inputs) {
            for (auto leaf = in.second->cbeginLeaf(); leaf; ++leaf) acc.touchLeaf(leaf->origin());
        }
    }

    BoolTreeT candidates(false);
    for (const auto& in : inputs) candidates.topologyUnion(*in.second);
    candidates.voxelizeActiveTiles();
    tools::dilateActiveValues(candidates, 1, NN_FACE, IGNORE_TILES);

    return build(*xform, lab, candidates, sampler, halfWidth, /*reseed=*/true);
}


////////////////////////////////////////


namespace lsmo_internal {

/// @brief Index of offset (i,j,k) in the 19-point layout of
/// math::CurvatureStencil (0 centre, 1-6 faces, 7-18 edges), or -1.
constexpr int
stencilIndex(int i, int j, int k)
{
    constexpr int offs[19][3] = {
        {0,0,0}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1},
        {-1,-1,0}, {1,-1,0}, {-1,1,0}, {1,1,0},
        {-1,0,-1}, {1,0,-1}, {-1,0,1}, {1,0,1},
        {0,-1,-1}, {0,1,-1}, {0,-1,1}, {0,1,1}};
    for (int n = 0; n < 19; ++n) {
        if (offs[n][0] == i && offs[n][1] == j && offs[n][2] == k) return n;
    }
    return -1;
}

} // namespace lsmo_internal


/// @brief Read-only 19-point stencil of psi_l = (L==l ? -phi : +phi) values
/// at time n (the main buffers).
/// @details The layout of the values is that of math::CurvatureStencil and
/// the curvature arithmetic is identical to
/// math::CurvatureStencil::meanCurvatureNormGrad. Voxels outside the active
/// band read the phi background with the sign given by their label.
/// All derivatives are in world units. Not thread-safe; use one per thread.
template<typename MultiLevelSetT>
class LabelAwareStencil
{
public:
    using FloatTreeT = typename MultiLevelSetT::FloatGridT::TreeType;
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;

    explicit LabelAwareStencil(const MultiLevelSetT& mls)
        : mTransform(mls.phi()->transform())
        , mPhiAcc(mls.phi()->tree())
        , mLabelAcc(mls.labels()->tree())
        , mDx(float(mls.voxelSize()))
        , mCenter(0, 0, 0)
    {
        for (float& v : mValues) v = 0.0f;
    }

    /// @brief Fill the stencil around @a ijk with psi of @a label.
    void moveTo(const Coord& ijk, int32_t label)
    {
        mCenter = ijk;
        mLabel = label;
        static const Coord offs[19] = {
            Coord(0,0,0), Coord(-1,0,0), Coord(1,0,0), Coord(0,-1,0), Coord(0,1,0),
            Coord(0,0,-1), Coord(0,0,1),
            Coord(-1,-1,0), Coord(1,-1,0), Coord(-1,1,0), Coord(1,1,0),
            Coord(-1,0,-1), Coord(1,0,-1), Coord(-1,0,1), Coord(1,0,1),
            Coord(0,-1,-1), Coord(0,1,-1), Coord(0,-1,1), Coord(0,1,1)};
        for (int n = 0; n < 19; ++n) {
            const Coord c = ijk + offs[n];
            const float d = mPhiAcc.getValue(c);
            mValues[n] = (static_cast<int32_t>(mLabelAcc.getValue(c)) == label) ? -d : d;
        }
    }

    const Coord& center() const { return mCenter; }
    int32_t label() const { return mLabel; }
    float voxelSize() const { return mDx; }
    const math::Transform& transform() const { return mTransform; }

    /// @brief psi at offset (i,j,k) from the centre; the 19 points have at
    /// most two non-zero offsets.
    template<int i, int j, int k>
    float getValue() const
    {
        static_assert(i >= -1 && i <= 1 && j >= -1 && j <= 1 && k >= -1 && k <= 1,
            "offsets must be in [-1,1]");
        constexpr int n = lsmo_internal::stencilIndex(i, j, k);
        static_assert(n >= 0, "corner offsets are not part of the 19-point stencil");
        return mValues[n];
    }

    /// @brief Central-difference gradient in world units.
    Vec3f gradient() const
    {
        const float inv2Dx = 0.5f / mDx;
        return Vec3f(mValues[2] - mValues[1],
                     mValues[4] - mValues[3],
                     mValues[6] - mValues[5]) * inv2Dx;
    }

    /// @brief One-sided differences in world units: dneg = (psi0 - psi-)/dx,
    /// dpos = (psi+ - psi0)/dx for x, y, z.
    void upwindDifferences(float dneg[3], float dpos[3]) const
    {
        const float invDx = 1.0f / mDx;
        for (int a = 0; a < 3; ++a) {
            dneg[a] = (mValues[0] - mValues[1 + 2 * a]) * invDx;
            dpos[a] = (mValues[2 + 2 * a] - mValues[0]) * invDx;
        }
    }

    /// @brief Mean curvature (k1+k2)/2 times |grad psi|, world units; 0 where
    /// the gradient vanishes. Same arithmetic as
    /// math::CurvatureStencil::meanCurvatureNormGrad.
    float meanCurvatureNormGrad() const
    {
        const double v[19] = {
            mValues[0], mValues[1], mValues[2], mValues[3], mValues[4], mValues[5],
            mValues[6], mValues[7], mValues[8], mValues[9], mValues[10], mValues[11],
            mValues[12], mValues[13], mValues[14], mValues[15], mValues[16],
            mValues[17], mValues[18]};
        const double Dx = 0.5 * (v[2] - v[1]), Dy = 0.5 * (v[4] - v[3]),
                     Dz = 0.5 * (v[6] - v[5]);
        const double Dx2 = Dx * Dx, Dy2 = Dy * Dy, Dz2 = Dz * Dz, normGrad2 = Dx2 + Dy2 + Dz2;
        if (normGrad2 <= math::Tolerance<double>::value()) return 0.0f;
        const double Dxx = v[2] - 2 * v[0] + v[1];
        const double Dyy = v[4] - 2 * v[0] + v[3];
        const double Dzz = v[6] - 2 * v[0] + v[5];
        const double Dxy = 0.25 * (v[10] - v[8] + v[7] - v[9]);
        const double Dxz = 0.25 * (v[14] - v[12] + v[11] - v[13]);
        const double Dyz = 0.25 * (v[18] - v[16] + v[15] - v[17]);
        const double alpha = Dx2 * (Dyy + Dzz) + Dy2 * (Dxx + Dzz) + Dz2 * (Dxx + Dyy)
            - 2 * (Dx * (Dy * Dxy + Dz * Dxz) + Dy * Dz * Dyz);
        const float inv2Dx = float(0.5 / mDx);
        const float invDx2 = 4.0f * inv2Dx * inv2Dx;
        return float(alpha * invDx2 / (2 * normGrad2));
    }

private:
    const math::Transform& mTransform;
    tree::ValueAccessor<const FloatTreeT> mPhiAcc;
    tree::ValueAccessor<const LabelTreeT> mLabelAcc;
    float mDx;
    Coord mCenter;
    int32_t mLabel = 0;
    float mValues[19];
};


/// @brief Built-in speed functor reproducing wedge's multi-object force
/// model: curvature + upwind pressure + upwind advection (spec 3.4).
/// @details Satisfies the speed functor concept of spec 3.3. Label 0 is
/// handled by the evolver as passive (rulings R3, R20), so isFixed(0) is
/// false here. Speeds are world-space; with unit voxel size they equal
/// wedge's index-space values. operator() is const and thread-safe.
template<typename MultiLevelSetT>
class MultiObjectForces
{
public:
    /// Curvature weight w: speed w*kappa*|grad psi| (clamped).
    void setCurvatureWeight(float w) { mCurvatureWeight = w; }
    void setMaxCurvatureForce(float m = 10.0f) { mMaxCurvatureForce = m; }

    /// @brief Pressure force field. A deep copy is rescaled once (wedge
    /// rescale()): values are shifted by @a target and the negative and
    /// positive parts scaled independently to [-1, 1] (scale 1 for a part
    /// whose extreme magnitude is at most 1e-4). A NaN @a target skips the
    /// rescale, as in wedge.
    /// @throw ValueError if @a pressure is null.
    void setPressure(FloatGrid::ConstPtr pressure, float weight,
        float target = std::numeric_limits<float>::quiet_NaN())
    {
        if (!pressure) OPENVDB_THROW(ValueError, "MultiObjectForces: null pressure grid");
        FloatGrid::Ptr copy = pressure->deepCopy();
        if (!std::isnan(target)) {
            FloatTree& tree = copy->tree();
            float minValue = 1e30f, maxValue = -1e30f;
            auto visit = [&](float x) {
                const float val = x - target;
                minValue = std::min(val, minValue);
                maxValue = std::max(val, maxValue);
            };
            visit(tree.background());
            for (auto it = tree.cbeginValueAll(); it; ++it) visit(*it);
            const float normMin = (std::abs(minValue) > 1e-4f) ? 1.0f / std::abs(minValue) : 1.0f;
            const float normMax = (std::abs(maxValue) > 1e-4f) ? 1.0f / std::abs(maxValue) : 1.0f;
            auto rescale = [&](float x) {
                const float val = x - target;
                return (val < 0.0f) ? val * normMin : val * normMax;
            };
            const float bg = rescale(tree.background());
            for (auto it = tree.beginValueAll(); it; ++it) it.setValue(rescale(*it));
            tree.root().setBackground(bg, /*updateChildNodes=*/false);
        }
        mPressure = copy;
        mPressureWeight = weight;
    }

    /// @brief Advection velocity (world units) sampled with BoxSampler.
    /// @throw ValueError if @a velocity is null.
    void setAdvection(Vec3SGrid::ConstPtr velocity, float weight)
    {
        if (!velocity) OPENVDB_THROW(ValueError, "MultiObjectForces: null advection grid");
        mAdvection = velocity;
        mAdvectionWeight = weight;
    }

    void setInactiveLabels(std::set<int32_t> labels) { mInactive = std::move(labels); }
    void setFixedLabels(std::set<int32_t> labels) { mFixed = std::move(labels); }

    /// The internal rescaled pressure copy (null if none was set).
    FloatGrid::ConstPtr pressure() const { return mPressure; }

    /// @brief dpsi_l/dt at @a ijk; the stencil must have been moved to
    /// (@a ijk, @a label).
    float operator()(const Coord& ijk, int32_t label,
        const LabelAwareStencil<MultiLevelSetT>& s) const
    {
        if (!isActive(label)) return 0.999f;
        if (mFixed.count(label)) return 0.0f;

        float dneg[3], dpos[3];
        s.upwindDifferences(dneg, dpos);

        float kappa = mCurvatureWeight * s.meanCurvatureNormGrad();
        kappa = std::max(-mMaxCurvatureForce, std::min(mMaxCurvatureForce, kappa));

        float advection = 0.0f;
        if (mAdvection) {
            const Vec3d w = s.transform().indexToWorld(ijk);
            const Vec3R idx = mAdvection->transform().worldToIndex(w);
            const Vec3f vel = tools::BoxSampler::sample(mAdvection->tree(), idx);
            for (int a = 0; a < 3; ++a) {
                const float f = mAdvectionWeight * vel[a];
                if (f > 0.0f) advection += f * dneg[a];
                else if (f < 0.0f) advection += f * dpos[a];
            }
        }

        float pressure = 0.0f;
        if (mPressure) {
            const Vec3d w = s.transform().indexToWorld(ijk);
            const Vec3R idx = mPressure->transform().worldToIndex(w);
            const float force = mPressureWeight * tools::BoxSampler::sample(mPressure->tree(), idx);
            float gsPos = 0.0f, gsNeg = 0.0f;
            for (int a = 0; a < 3; ++a) {
                const float nMax = std::max(dneg[a], 0.0f), nMin = std::min(dneg[a], 0.0f);
                const float pMax = std::max(dpos[a], 0.0f), pMin = std::min(dpos[a], 0.0f);
                gsPos += nMax * nMax + pMin * pMin;
                gsNeg += pMax * pMax + nMin * nMin;
            }
            if (force > 0.0f) pressure = -force * std::sqrt(gsPos);
            else if (force < 0.0f) pressure = -force * std::sqrt(gsNeg);
        }
        return -advection + kappa + pressure;
    }

    /// True for labels declared fixed. Label 0 is NOT fixed here: the
    /// evolver treats it as passive and never asks the functor (R3, R20).
    bool isFixed(int32_t label) const { return label != 0 && mFixed.count(label) != 0; }
    bool isActive(int32_t label) const { return mInactive.count(label) == 0; }

    /// dx^2/(6 w) for curvature weight w > 0, else +infinity.
    float stableDt(float dx) const
    {
        return mCurvatureWeight > 0.0f ? dx * dx / (6.0f * mCurvatureWeight)
                                       : std::numeric_limits<float>::infinity();
    }

private:
    float mCurvatureWeight = 0.0f;
    float mMaxCurvatureForce = 10.0f;
    FloatGrid::ConstPtr mPressure;
    float mPressureWeight = 0.0f;
    Vec3SGrid::ConstPtr mAdvection;
    float mAdvectionWeight = 0.0f;
    std::set<int32_t> mInactive, mFixed;
};


////////////////////////////////////////


/// @brief Multi-object level set evolver (MOGAC / MCLS, spec section 4).
///
/// @details One forward Euler step of step() does:
/// -# <b>Candidates and speeds.</b> Every active voxel that is not of a fixed
///    label and has phi <= 0.5*dx or a face neighbour (active or not) of a
///    different label is updated. Its candidates are its own label and the
///    labels of its face neighbours, minus fixed labels (rulings R3, R21).
///    Label 0 is passive: it is a candidate only if it is the voxel's or a
///    face neighbour's label or the voxel touches a fixed label, and its
///    value is the complement v_0 = -min v_l over the non-zero candidates,
///    so object/background interfaces move at the object's full speed. The
///    speed functor is called (on time-n data through a LabelAwareStencil)
///    only for non-zero candidates; labels with isActive() false get
///    +0.999 (R4).
/// -# <b>Time step.</b> dt = min(maxDt, cfl*dx/max|F|, speed.stableDt(dx));
///    with setClampSpeed(true), F is clamped to [-1, 1] and
///    dt = min(maxDt, speed.stableDt(dx)).
/// -# <b>Projection.</b> v_l = psi_l + dt*F_l with psi_l = -phi for the own
///    label and +phi otherwise; the new label is the smallest v (ties to the
///    smaller label) and phi = |v_1 - v_2|/2 over the two smallest values, or
///    |v_1| with one candidate. Results are written to copies of both trees
///    (time-n data stays readable until both are swapped in together).
///    A voxel whose face neighbour changed label is (re)projected with that
///    new label added to its candidates (still evaluated on time-n data),
///    repeated until no label changes: the redistancer freezes every voxel
///    next to a label change (ruling R12), so each such voxel must carry a
///    projected value. Speeds first evaluated here are bounded by the
///    max|F| that set dt.
/// -# <b>Orphan cleanup</b> (4.5): a voxel whose label occurs nowhere in its
///    26-neighbourhood takes the label and phi of its smallest-phi active
///    neighbour (fixed labels excluded).
/// -# <b>Band maintenance</b> (4.6): both trees are swapped in; inactive face
///    neighbours of active voxels with phi < (halfWidth-1)*dx are activated
///    (phi = background, label of the smallest-phi active face neighbour);
///    MultiLevelSetRedistance::redistance() (interface frozen); voxels of
///    fixed labels get their time-n phi back; voxels with
///    phi >= halfWidth*dx are deactivated in both trees (labels kept, R7b);
///    pruneInactive(). Debug builds assert validate().
///
/// Voxels of fixed labels never change label or phi and no label can
/// capture them; a step with no non-zero candidate anywhere changes nothing.
///
/// @par SpeedT concept (spec 3.3)
/// @code
/// float operator()(const Coord& ijk, int32_t label,
///                  const LabelAwareStencil<MultiLevelSetT>& s) const; // dpsi/dt
/// bool  isFixed(int32_t label) const;
/// bool  isActive(int32_t label) const;
/// float stableDt(float dx) const; // +inf if none
/// @endcode
/// operator() must be thread-safe. All quantities are in world units.
///
/// @par Example
/// @code
/// auto mls = tools::MultiLevelSet<>::fromLabelGrid(*labels);
/// tools::MultiObjectForces<tools::MultiLevelSet<>> forces;
/// forces.setCurvatureWeight(0.5f);
/// tools::LevelSetMultiObject<tools::MultiLevelSet<>,
///     tools::MultiObjectForces<tools::MultiLevelSet<>>> evolver(*mls, forces);
/// evolver.advect(0.0, 10.0);
/// @endcode
template<typename MultiLevelSetT, typename SpeedT, typename InterruptT = util::NullInterrupter>
class LevelSetMultiObject
{
public:
    using FloatTreeT = typename MultiLevelSetT::FloatGridT::TreeType;
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;
    using LabelT = typename LabelTreeT::ValueType;
    using StencilT = LabelAwareStencil<MultiLevelSetT>;

    /// @a mls and @a speed must outlive the evolver.
    LevelSetMultiObject(MultiLevelSetT& mls, const SpeedT& speed, InterruptT* interrupt = nullptr)
        : mMls(mls), mSpeed(speed), mInterrupter(interrupt)
        , mTolerance(float(1e-4 * mls.voxelSize()))
    {
    }

    /// @brief CFL number (default 0.999).
    /// @throw ValueError if @a cfl is not positive.
    void setCFL(float cfl)
    {
        if (!(cfl > 0.0f)) OPENVDB_THROW(ValueError, "LevelSetMultiObject: CFL must be positive");
        mCfl = cfl;
    }
    /// @brief Clamp speeds to [-1, 1] and always take dt = maxDt (capped by
    /// speed.stableDt), as wedge's clampSpeed.
    void setClampSpeed(bool on) { mClampSpeed = on; }
    /// @brief TBB grain size in leaves; zero or negative runs serially.
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }
    /// @brief Redistance tolerance in world units (default 1e-4*dx).
    void setRedistanceTolerance(float tolWorld) { mTolerance = tolWorld; }

    float cfl() const { return mCfl; }
    bool clampSpeed() const { return mClampSpeed; }
    int grainSize() const { return mGrainSize; }
    float redistanceTolerance() const { return mTolerance; }

    /// @brief One forward Euler step of at most @a maxDt (world time).
    /// @return the time step taken; @a maxDt (or speed.stableDt if smaller)
    /// without changes if nothing can move, 0 if @a maxDt <= 0.
    float step(float maxDt);

    /// @brief Step from @a t0 to @a t1, checking the interrupter before each
    /// step. On interrupt the state after the last completed step is kept
    /// (it is valid).
    /// @return the number of steps taken.
    size_t advect(double t0, double t1)
    {
        if (mInterrupter) mInterrupter->start("Evolving multi-object level set");
        size_t n = 0;
        double t = t0;
        const double eps = 1e-6 * std::max(1.0, std::abs(t1));
        while (t1 - t > eps) {
            if (util::wasInterrupted(mInterrupter)) break;
            const float dt = this->step(float(t1 - t));
            if (!(dt > 0.0f)) break;
            t += dt;
            ++n;
        }
        if (mInterrupter) mInterrupter->end();
        return n;
    }

private:
    /// Own label + 6 face labels at time n + 6 face labels at time n+1.
    static constexpr int kMaxCandidates = 13;

    /// Candidate list of one updated voxel: non-zero labels with their
    /// speeds, and whether passive label 0 participates.
    struct VoxelEntry
    {
        Index offset = 0;
        int32_t own = 0;
        float phi = 0.0f; // time n
        int n = 0;
        bool zero = false;
        std::array<std::pair<int32_t, float>, kMaxCandidates> cand;

        bool has(int32_t l) const
        {
            for (int i = 0; i < n; ++i) if (cand[i].first == l) return true;
            return false;
        }
    };

    struct LeafData
    {
        std::vector<VoxelEntry> entries;
        std::vector<int32_t> slot; // voxel offset -> entry index, or -1
        std::vector<Coord> changed;
        float maxSpeed = 0.0f;
        bool any = false;

        VoxelEntry& add(Index offset)
        {
            if (slot.empty()) slot.assign(FloatTreeT::LeafNodeType::NUM_VALUES, -1);
            slot[offset] = int32_t(entries.size());
            entries.emplace_back();
            entries.back().offset = offset;
            return entries.back();
        }
        VoxelEntry* find(Index offset)
        {
            if (slot.empty() || slot[offset] < 0) return nullptr;
            return &entries[size_t(slot[offset])];
        }
    };

    template<typename BodyT>
    void forEachIndex(size_t count, const BodyT& body) const
    {
        if (mGrainSize > 0) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, size_t(mGrainSize)),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) body(i);
                });
        } else {
            for (size_t i = 0; i < count; ++i) body(i);
        }
    }

    float evalSpeed(StencilT& s, const Coord& ijk, int32_t label) const
    {
        float f;
        if (!mSpeed.isActive(label)) {
            f = 0.999f; // R4
        } else {
            s.moveTo(ijk, label);
            f = mSpeed(ijk, label, s);
        }
        if (mClampSpeed) f = std::max(-1.0f, std::min(1.0f, f));
        return f;
    }

    /// Add @a l to @a e's candidates (R21): label 0 and fixed labels only
    /// make passive label 0 participate. A new non-zero label gets a NaN
    /// speed (not yet evaluated). Returns true if the candidate set grew.
    bool addCandidate(VoxelEntry& e, int32_t l) const
    {
        if (l == 0 || mSpeed.isFixed(l)) { // label 0, or a wall: passive 0
            const bool grew = !e.zero;
            e.zero = true;
            return grew;
        }
        if (e.has(l) || e.n >= kMaxCandidates) return false;
        e.cand[e.n++] = std::make_pair(l, std::numeric_limits<float>::quiet_NaN());
        return true;
    }

    /// Evaluate the NaN speeds of @a e at @a ijk; with @a bound > 0 they are
    /// clamped to [-bound, bound]. Returns the max |speed| evaluated.
    float evalSpeeds(StencilT& s, const Coord& ijk, VoxelEntry& e, float bound) const
    {
        float mx = 0.0f;
        for (int c = 0; c < e.n; ++c) {
            if (!std::isnan(e.cand[c].second)) continue;
            float f = this->evalSpeed(s, ijk, e.cand[c].first);
            if (bound > 0.0f) f = std::max(-bound, std::min(bound, f));
            e.cand[c].second = f;
            mx = std::max(mx, std::abs(f));
        }
        return mx;
    }

    /// Two-smallest projection; false if @a e has no non-zero candidate.
    static bool project(const VoxelEntry& e, float dt, int32_t& label, float& phi)
    {
        if (e.n == 0) return false;
        const float inf = std::numeric_limits<float>::infinity();
        float v1 = inf, v2 = inf, minV = inf;
        int32_t l1 = 0, l2 = 0;
        int count = 0;
        auto consider = [&](int32_t l, float v) {
            ++count;
            if (count == 1 || v < v1 || (v == v1 && l < l1)) {
                v2 = v1; l2 = l1; v1 = v; l1 = l;
            } else if (count == 2 || v < v2 || (v == v2 && l < l2)) {
                v2 = v; l2 = l;
            }
        };
        for (int i = 0; i < e.n; ++i) {
            const int32_t l = e.cand[i].first;
            const float psi = (l == e.own) ? -e.phi : e.phi;
            const float v = psi + dt * e.cand[i].second;
            consider(l, v);
            minV = std::min(minV, v);
        }
        if (e.zero) consider(0, -minV); // passive complement (R3)
        label = l1;
        phi = (count > 1) ? std::abs(0.5f * (v1 - v2)) : std::abs(v1);
        return true;
    }

    /// Replace both trees together (time n -> n+1).
    void swapBuffers(typename FloatTreeT::Ptr phi, typename LabelTreeT::Ptr labels)
    {
        mMls.phi()->setTree(phi);
        mMls.labels()->setTree(labels);
    }

    void propagate(const tree::LeafManager<const FloatTreeT>& leafs, std::vector<LeafData>& data,
        FloatTreeT& newPhi, LabelTreeT& newLabels, float dt, float bound);
    void cleanOrphans(FloatTreeT& newPhi, LabelTreeT& newLabels);
    void bandMaintenance(const FloatTreeT& oldPhi);

    MultiLevelSetT& mMls;
    const SpeedT& mSpeed;
    InterruptT* mInterrupter;
    float mCfl = 0.999f;
    bool mClampSpeed = false;
    int mGrainSize = 1;
    float mTolerance;
};


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::step(float maxDt)
{
    if (!(maxDt > 0.0f)) return 0.0f;
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();
    const double dx = mMls.voxelSize();
    const float halfDx = float(0.5 * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // The label tree needs a leaf wherever phi has one (normally a no-op).
    {
        tree::ValueAccessor<LabelTreeT> acc(labels);
        for (auto leaf = phi.cbeginLeaf(); leaf; ++leaf) acc.touchLeaf(leaf->origin());
    }

    // 1-2. interface set, candidates and speeds, all on time-n data
    tree::LeafManager<const FloatTreeT> leafs(phi);
    const size_t leafCount = leafs.leafCount();
    std::vector<LeafData> data(leafCount);
    forEachIndex(leafCount, [&](size_t i) {
        const auto& leaf = leafs.leaf(i);
        LeafData& d = data[i];
        StencilT stencil(mMls);
        tree::ValueAccessor<const LabelTreeT> lab(labels);
        for (auto it = leaf.cbeginValueOn(); it; ++it) {
            const Coord p = it.getCoord();
            const int32_t own = static_cast<int32_t>(lab.getValue(p));
            if (own != 0 && mSpeed.isFixed(own)) continue; // walls are skipped
            int32_t nb[6];
            bool seed = false;
            for (int f = 0; f < 6; ++f) {
                nb[f] = static_cast<int32_t>(lab.getValue(p + offs[f]));
                seed |= nb[f] != own;
            }
            if (!seed && *it > halfDx) continue;
            VoxelEntry& e = d.add(it.pos());
            e.own = own;
            e.phi = *it;
            this->addCandidate(e, own);
            for (int f = 0; f < 6; ++f) this->addCandidate(e, nb[f]);
            d.maxSpeed = std::max(d.maxSpeed, this->evalSpeeds(stencil, p, e, 0.0f));
            d.any |= e.n > 0;
        }
    });

    float maxSpeed = 0.0f;
    bool any = false;
    for (const LeafData& d : data) { maxSpeed = std::max(maxSpeed, d.maxSpeed); any |= d.any; }

    // 3. time step
    float dt = std::min(maxDt, mSpeed.stableDt(float(dx)));
    if (!any) return dt; // no non-zero candidate anywhere: nothing moves
    if (!mClampSpeed && maxSpeed > 0.0f) dt = std::min(dt, float(mCfl * dx / maxSpeed));

    // 4. two-smallest projection, written to copies of both trees
    typename FloatTreeT::Ptr newPhi(new FloatTreeT(phi));
    typename LabelTreeT::Ptr newLabels(new LabelTreeT(labels));
    forEachIndex(leafCount, [&](size_t i) {
        LeafData& d = data[i];
        if (d.entries.empty()) return;
        const auto& leaf = leafs.leaf(i);
        auto* phiLeaf = newPhi->probeLeaf(leaf.origin());
        auto* labLeaf = newLabels->probeLeaf(leaf.origin());
        for (const VoxelEntry& e : d.entries) {
            int32_t l; float v;
            if (!project(e, dt, l, v)) continue;
            phiLeaf->setValueOnly(e.offset, v);
            labLeaf->setValueOnly(e.offset, LabelT(l));
            if (l != e.own) d.changed.push_back(leaf.offsetToGlobalCoord(e.offset));
        }
    });
    this->propagate(leafs, data, *newPhi, *newLabels, dt,
        (mClampSpeed || maxSpeed <= 0.0f) ? 1.0f : maxSpeed);

    // 5. orphan cleanup
    this->cleanOrphans(*newPhi, *newLabels);

    // 6a. swap both trees together
    typename FloatTreeT::Ptr oldPhi = mMls.phi()->treePtr();
    this->swapBuffers(newPhi, newLabels);

    // 6b-e.
    this->bandMaintenance(*oldPhi);
#ifndef NDEBUG
    assert(mMls.validate());
#endif
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::propagate(
    const tree::LeafManager<const FloatTreeT>& leafs, std::vector<LeafData>& data,
    FloatTreeT& newPhi, LabelTreeT& newLabels, float dt, float bound)
{
    using LeafT = typename FloatTreeT::LeafNodeType;
    const Coord* offs = lsmo_internal::faceOffsets();
    std::vector<Coord> front;
    for (LeafData& d : data) {
        front.insert(front.end(), d.changed.begin(), d.changed.end());
        d.changed.clear();
    }
    if (front.empty()) return;

    std::map<Coord, size_t> leafIndex;
    for (size_t i = 0; i < leafs.leafCount(); ++i) leafIndex[leafs.leaf(i).origin()] = i;
    StencilT stencil(mMls);
    tree::ValueAccessor<const FloatTreeT> phiAcc(mMls.phi()->tree());
    tree::ValueAccessor<const LabelTreeT> labAcc(mMls.labels()->tree());
    tree::ValueAccessor<FloatTreeT> newPhiAcc(newPhi);
    tree::ValueAccessor<LabelTreeT> newLabAcc(newLabels);

    while (!front.empty()) {
        std::vector<Coord> next;
        for (const Coord& c : front) {
            for (int f = 0; f < 6; ++f) {
                const Coord q = c + offs[f];
                float phiQ;
                if (!phiAcc.probeValue(q, phiQ)) continue; // only active voxels
                const int32_t own = static_cast<int32_t>(labAcc.getValue(q));
                if (own != 0 && mSpeed.isFixed(own)) continue;
                const auto found = leafIndex.find(q & ~(LeafT::DIM - 1));
                if (found == leafIndex.end()) continue;
                LeafData& d = data[found->second];
                const Index offset = LeafT::coordToOffset(q);
                VoxelEntry* e = d.find(offset);
                bool grew = false;
                if (!e) {
                    e = &d.add(offset);
                    e->own = own;
                    e->phi = phiQ;
                    this->addCandidate(*e, own);
                    for (int g = 0; g < 6; ++g) {
                        this->addCandidate(*e, static_cast<int32_t>(labAcc.getValue(q + offs[g])));
                    }
                    grew = true;
                }
                for (int g = 0; g < 6; ++g) {
                    grew |= this->addCandidate(*e,
                        static_cast<int32_t>(newLabAcc.getValue(q + offs[g])));
                }
                if (!grew) continue;
                this->evalSpeeds(stencil, q, *e, bound);
                int32_t l; float v;
                if (!project(*e, dt, l, v)) continue;
                const int32_t cur = static_cast<int32_t>(newLabAcc.getValue(q));
                newPhiAcc.setValueOnly(q, v);
                newLabAcc.setValueOnly(q, LabelT(l));
                if (l != cur) next.push_back(q);
            }
        }
        front.swap(next);
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::cleanOrphans(
    FloatTreeT& newPhi, LabelTreeT& newLabels)
{
    struct Fix { Coord ijk; LabelT label; float phi; };
    tree::LeafManager<const FloatTreeT> leafs(newPhi);
    std::vector<std::vector<Fix>> fixes(leafs.leafCount());
    forEachIndex(leafs.leafCount(), [&](size_t i) {
        tree::ValueAccessor<const FloatTreeT> phiAcc(newPhi);
        tree::ValueAccessor<const LabelTreeT> labAcc(newLabels);
        for (auto it = leafs.leaf(i).cbeginValueOn(); it; ++it) {
            const Coord p = it.getCoord();
            const LabelT own = labAcc.getValue(p);
            if (own != LabelT(0) && mSpeed.isFixed(static_cast<int32_t>(own))) continue;
            bool found = false;
            float best = std::numeric_limits<float>::infinity();
            LabelT bestLabel = own;
            for (int a = -1; a <= 1 && !found; ++a) {
                for (int b = -1; b <= 1 && !found; ++b) {
                    for (int c = -1; c <= 1 && !found; ++c) {
                        if (a == 0 && b == 0 && c == 0) continue;
                        const Coord q = p.offsetBy(a, b, c);
                        const LabelT l = labAcc.getValue(q);
                        if (l == own) { found = true; break; }
                        float v;
                        if (!phiAcc.probeValue(q, v)) continue;
                        if (l != LabelT(0) && mSpeed.isFixed(static_cast<int32_t>(l))) continue;
                        if (v < best || (v == best && l < bestLabel)) { best = v; bestLabel = l; }
                    }
                }
            }
            if (!found && bestLabel != own) fixes[i].push_back(Fix{p, bestLabel, best});
        }
    });
    tree::ValueAccessor<FloatTreeT> phiAcc(newPhi);
    tree::ValueAccessor<LabelTreeT> labAcc(newLabels);
    for (const auto& list : fixes) {
        for (const Fix& fx : list) {
            phiAcc.setValueOnly(fx.ijk, fx.phi);
            labAcc.setValueOnly(fx.ijk, fx.label);
        }
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::bandMaintenance(const FloatTreeT& oldPhi)
{
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();
    const double dx = mMls.voxelSize();
    const float bg = phi.background();
    const float grow = float((mMls.halfWidth() - 1) * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // b. activate the inactive face neighbours of voxels with
    // phi < (halfWidth-1)*dx; label of the smallest-phi active face neighbour
    {
        std::map<Coord, std::pair<float, LabelT>> added;
        tree::ValueAccessor<const FloatTreeT> phiAcc(phi);
        tree::ValueAccessor<const LabelTreeT> labAcc(labels);
        for (auto it = phi.cbeginValueOn(); it; ++it) {
            if (!(*it < grow)) continue;
            const Coord p = it.getCoord();
            for (int f = 0; f < 6; ++f) {
                const Coord q = p + offs[f];
                if (phiAcc.isValueOn(q)) continue;
                const LabelT l = labAcc.getValue(p);
                auto ins = added.insert(std::make_pair(q, std::make_pair(*it, l)));
                if (!ins.second) {
                    auto& cur = ins.first->second;
                    if (*it < cur.first || (*it == cur.first && l < cur.second)) {
                        cur = std::make_pair(*it, l);
                    }
                }
            }
        }
        tree::ValueAccessor<FloatTreeT> phiW(phi);
        tree::ValueAccessor<LabelTreeT> labW(labels);
        for (const auto& kv : added) {
            phiW.setValueOn(kv.first, bg);
            labW.setValueOn(kv.first, kv.second.second);
        }
    }

    // c. redistance with the interface frozen (R12)
    {
        MultiLevelSetRedistance<MultiLevelSetT> redist(mMls);
        redist.setTolerance(mTolerance);
        redist.setGrainSize(mGrainSize);
        redist.redistance();
    }

    // Fixed-label voxels keep their time-n phi; d. deactivate
    // phi >= halfWidth*dx in both trees (labels kept, R7b).
    {
        tree::LeafManager<FloatTreeT> leafs(phi);
        forEachIndex(leafs.leafCount(), [&](size_t i) {
            auto& leaf = leafs.leaf(i);
            auto* labLeaf = labels.probeLeaf(leaf.origin());
            const auto* oldLeaf = oldPhi.probeConstLeaf(leaf.origin());
            for (auto it = leaf.beginValueOn(); it; ++it) {
                const LabelT l = labLeaf ? labLeaf->getValue(it.pos()) : labels.getValue(it.getCoord());
                if (oldLeaf && oldLeaf->isValueOn(it.pos()) && l != LabelT(0)
                    && mSpeed.isFixed(static_cast<int32_t>(l))) {
                    it.setValue(oldLeaf->getValue(it.pos()));
                }
                if (*it < bg) continue;
                leaf.setValueOff(it.pos(), bg);
                if (labLeaf) labLeaf->setActiveState(it.pos(), false);
            }
        });
    }
    mMls.pruneInactive();
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
