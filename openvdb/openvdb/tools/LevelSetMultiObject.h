// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObject.h
///
/// @brief Multi-object level sets: a single unsigned distance grid phi
/// paired with an integer grid of object labels (MOGAC). This header
/// currently provides the MultiLevelSet data holder (with its factories
/// fromLabelGrid and fromSignedGrids) and the second-order label-aware
/// redistancer MultiLevelSetRedistance.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/NullInterrupter.h>
#include "Morphology.h"// for dilateActiveValues, erodeActiveValues
#include "Prune.h"// for pruneInactive

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    using LeafManagerT = tree::LeafManager<FloatTreeT>;

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
        const float bg = phi.background();
        const int maxIter = int(std::ceil(mMls.halfWidth() * std::sqrt(3.0))) + 2;
        const bool serial = mGrainSize <= 0;

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

        // 5.2: reset non-seed voxels, then Jacobi with one aux buffer.
        LeafManagerT leafs(phi, 1, serial);
        this->forEachLeaf(leafs, [&](typename LeafManagerT::LeafRange::Iterator& leafIt) {
            const auto* mask = seeds.probeConstLeaf(leafIt->origin());
            for (auto it = leafIt->beginValueOn(); it; ++it) {
                if (!mask || !mask->isValueOn(it.pos())) it.setValue(kUnknown);
            }
        });
        leafs.syncAllBuffers(serial);

        int iter = 0;
        while (iter < maxIter) {
            if (util::wasInterrupted(mInterrupter)) break;
            ++iter;
            const float maxDelta = this->sweep(leafs, seeds, dx);
            leafs.swapLeafBuffer(1, serial);
            if (maxDelta < mTolerance) break;
        }

        // Unreached voxels (no seed nearby, or interrupted) get the background.
        this->forEachLeaf(leafs, [&](typename LeafManagerT::LeafRange::Iterator& leafIt) {
            for (auto it = leafIt->beginValueOn(); it; ++it) {
                if (*it == kUnknown) it.setValue(bg);
            }
        });

        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    static constexpr float kUnknown = std::numeric_limits<float>::max();

    template<typename OpT>
    void forEachLeaf(LeafManagerT& leafs, const OpT& op) const
    {
        auto body = [&](const typename LeafManagerT::LeafRange& range) {
            for (auto leafIt = range.begin(); leafIt; ++leafIt) op(leafIt);
        };
        if (mGrainSize > 0) tbb::parallel_for(leafs.leafRange(size_t(mGrainSize)), body);
        else body(leafs.leafRange());
    }

    /// One Jacobi iteration: read main buffers, write aux buffer 1.
    /// @return max|delta phi| over updated voxels.
    float sweep(LeafManagerT& leafs, const BoolTreeT& seeds, double dx) const
    {
        const FloatTreeT& phi = leafs.tree();
        const LabelTreeT& labels = mMls.labels()->tree();
        const Coord* offs = lsmo_internal::faceOffsets();

        auto body = [&](const typename LeafManagerT::LeafRange& range, float maxDelta) {
            tree::ValueAccessor<const FloatTreeT> acc(phi);
            tree::ValueAccessor<const LabelTreeT> labAcc(labels);
            for (auto leafIt = range.begin(); leafIt; ++leafIt) {
                auto& aux = leafIt.buffer(1);
                const auto* mask = seeds.probeConstLeaf(leafIt->origin());
                for (auto it = leafIt->cbeginValueOn(); it; ++it) {
                    if (mask && mask->isValueOn(it.pos())) continue;
                    const Coord p = it.getCoord();
                    const float old = *it;
                    const auto label = labAcc.getValue(p);
                    double abar[3], coef[3];
                    int n = 0;
                    for (int axis = 0; axis < 3; ++axis) {
                        float a = kUnknown, a2 = kUnknown;
                        for (int side = 0; side < 2; ++side) {
                            float v;
                            const Coord q = p + offs[2 * axis + side];
                            if (!acc.probeValue(q, v) || !(v < a)) continue;
                            a = v;
                            // a' must lie on p's side of the interface (same label)
                            const Coord q2 = q + offs[2 * axis + side];
                            if (!acc.probeValue(q2, a2) || labAcc.getValue(q2) != label) {
                                a2 = kUnknown;
                            }
                        }
                        if (a == kUnknown) continue;
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
        if (mGrainSize > 0) {
            return tbb::parallel_reduce(leafs.leafRange(size_t(mGrainSize)), 0.0f, body,
                [](float a, float b) { return std::max(a, b); });
        }
        return body(leafs.leafRange(), 0.0f);
    }

    /// Solve sum_i c_i (phi - abar_i)^2 = dx^2 over the sorted upwind set,
    /// dropping terms with phi <= abar_i.
    static double solveGodunov(double* abar, double* coef, int n, double dx)
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

    MultiLevelSetT& mMls;
    InterruptT* mInterrupter;
    float mTolerance;
    int mGrainSize = 1;
};


////////////////////////////////////////


namespace lsmo_internal {

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

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
