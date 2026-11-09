// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObject.h
///
/// @brief Multi-object level sets: a single unsigned distance grid phi
/// paired with an integer grid of object labels (MOGAC). This header
/// currently provides the MultiLevelSet data holder and the second-order
/// label-aware redistancer MultiLevelSetRedistance.

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

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <string>
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
/// q (active or not). Along each axis the nearest crossing lies at
/// theta*dx with theta = phi_p / (phi_p + phi_q) (theta = 0.5 if the sum is
/// below 1e-6*dx); the seed value is (sum_i 1/d_i^2)^(-1/2). All phi values
/// are read from @a snapshot (inactive neighbours read as its background),
/// so the result does not depend on evaluation order. Seed values are
/// written to @a out and flagged as active voxels of @a seedMask; nothing
/// else in @a out changes. Active tiles of @a out are ignored.
/// @param differs  copied once per task; called as differs(p, q).
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp>
void seedInterface(const FloatTreeT& snapshot, const DiffersOp& differs,
    FloatTreeT& out, BoolTreeT& seedMask, double dx, int grainSize)
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
                const double phiP = acc.getValue(p);
                double invSum = 0.0;
                bool seed = false, onInterface = false;
                for (int axis = 0; axis < 3; ++axis) {
                    double d = std::numeric_limits<double>::max();
                    for (int side = 0; side < 2; ++side) {
                        const Coord q = p + offs[2 * axis + side];
                        if (!op(p, q)) continue;
                        float phiQf;
                        if (!acc.probeValue(q, phiQf)) phiQf = bg;
                        const double sum = phiP + double(phiQf);
                        const double theta = (sum < eps) ? 0.5 : phiP / sum;
                        d = std::min(d, theta * dx);
                        seed = true;
                    }
                    if (d == std::numeric_limits<double>::max()) continue;
                    if (d <= 0.0) { onInterface = true; continue; }
                    invSum += 1.0 / (d * d);
                }
                if (!seed) continue;
                it.setValue(onInterface ? 0.0f : float(1.0 / std::sqrt(invSum)));
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
///    not, of a different label) are seeded with a subcell distance computed
///    from a snapshot of phi, and frozen.
/// -# All other active voxels solve |grad phi| = 1 by parallel Jacobi
///    iteration, using second-order one-sided upwind differences (Chopp 2001)
///    where two same-side upwind values are available (the outer one with
///    p's label and no larger than the inner one) and first order otherwise. Inactive neighbours are not used. Iteration stops when
///    max|delta phi| falls below the tolerance or after
///    ceil(halfWidth*sqrt(3)) + 2 iterations.
///
/// Only active phi voxels are written; topology, labels and inactive values
/// are untouched. Voxels with no path to a seed get the phi background.
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

    /// @brief Redistance phi in place.
    /// @return the number of Jacobi iterations performed.
    int redistance()
    {
        FloatTreeT& phi = mMls.phi()->tree();
        const double dx = mMls.voxelSize();
        const float bg = phi.background();
        const int maxIter = int(std::ceil(mMls.halfWidth() * std::sqrt(3.0))) + 2;
        const bool serial = mGrainSize <= 0;

        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");

        // 5.1: seeds from a read-only snapshot.
        BoolTreeT seeds(false);
        {
            const FloatTreeT snapshot(phi);
            lsmo_internal::seedInterface(snapshot,
                lsmo_internal::LabelDiffers<LabelTreeT>(mMls.labels()->tree()),
                phi, seeds, dx, mGrainSize);
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

    /// @brief Interface seeding only (spec 5.1), for per-label reconstruction.
    /// @details For each active voxel p of @a out, every face neighbour q with
    /// isInside(p) != isInside(q) defines a crossing; phi values come from
    /// @a phiSnapshot. Seeds are written to @a out and marked active in
    /// @a seedMask (which is overwritten); other voxels of @a out are unchanged.
    /// @param voxelSize world-space voxel size dx (trees carry no transform,
    /// and the subcell crossing offset theta*dx needs it).
    template<typename IsInsideOp>
    static void seedInterface(const FloatTreeT& phiSnapshot, IsInsideOp isInside,
        FloatTreeT& out, BoolTreeT& seedMask, double voxelSize)
    {
        lsmo_internal::seedInterface(phiSnapshot,
            lsmo_internal::InsideDiffers<IsInsideOp>(isInside), out, seedMask, voxelSize, 1);
    }

private:
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
                        if (a2 <= a) { // second order (Chopp 2001)
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

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
