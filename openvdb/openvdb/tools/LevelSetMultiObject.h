// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObject.h
///
/// @brief Multi-object level sets: evolution of several disjoint objects
/// (and their contacts) by multi-object geodesic active contours (MOGAC)
/// on a single unsigned distance grid paired with an integer label grid.
///
/// @details Provides the MultiLevelSet data holder (factories fromLabelGrid
/// and fromSignedGrids, and the second-order per-label signed distance
/// extractSignedDistance), the label-aware redistancer MultiLevelSetRedistance,
/// the LabelAwareStencil, the built-in speed functor MultiObjectForces
/// (advection, curvature, ...), the evolver LevelSetMultiObject (with opt-in
/// topology-preserving evolution, see setTopologyPreservingLabels, and
/// MultiObjectConnectivity), and, in LevelSetMultiObjectMesh.h, the
/// topology-constrained multi-label contouring multiObjectToMesh.
///
/// @par Example
/// @code
/// // signed distance grids (GRID_LEVEL_SET), one per non-zero label
/// std::map<int32_t, FloatGrid::ConstPtr> grids = { {1, sphereA}, {2, sphereB} };
/// auto mls = tools::MultiLevelSet<>::fromSignedGrids(grids);
///
/// // speed: advection by a velocity grid plus mean curvature smoothing
/// using MLS = tools::MultiLevelSet<>;
/// tools::MultiObjectForces<MLS> forces;
/// forces.setAdvection(velocityGrid, 1.0f);
/// forces.setCurvatureWeight(0.01f);
///
/// // evolve from t = 0 to t = 10
/// tools::LevelSetMultiObject<MLS, tools::MultiObjectForces<MLS>> evolver(*mls, forces);
/// // optional: keep these labels' digital topology (see MultiObjectConnectivity)
/// // evolver.setTopologyPreservingLabels({1, 2});
/// evolver.advect(0.0, 10.0);
///
/// // a regular signed distance grid for one label ...
/// FloatGrid::Ptr psi1 = mls->extractSignedDistance(1);
/// // ... or closed manifold meshes for all labels at once
/// std::map<int32_t, tools::LabelMesh> meshes = tools::multiObjectToMesh(*mls);
/// @endcode
///
/// @par References
/// - G. Russo, P. Smereka, "A remark on computing distance functions",
///   J. Comput. Phys. 163(1), 2000.
/// - F. du Chene, C. Min, F. Gibou, "Second-order accurate computation of
///   curvatures in a level set framework using novel high-order reinitialization
///   methods", J. Sci. Comput. 35, 2008. (Higher-order reinitialization.)
/// - D. L. Chopp, "Some improvements of the fast marching method",
///   SIAM J. Sci. Comput. 23(1), 2001.
/// - G. Bertrand, G. Malandain, "A new characterization of three-dimensional
///   simple points", Pattern Recognition Letters 15(2), 1994 (topological
///   numbers, used for topology preservation).
/// - J.-O. Lachaud, A. Montanvert, "Continuous analogs of digital boundaries:
///   a topological approach to iso-surfaces", Graphical Models 62(3), 2000.
/// - B. C. Lucas, M. Kazhdan, R. H. Taylor, "Multi-object geodesic active
///   contours (MOGAC)".
/// - B. C. Lucas, M. Kazhdan, R. H. Taylor, "Multi-object spring level sets
///   (MUSCLE)".

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
#include <openvdb/util/NodeMasks.h>// for FindLowestOn
#include <openvdb/util/Assert.h>
#include <openvdb/util/NullInterrupter.h>
#include "Morphology.h"// for dilateActiveValues, erodeActiveValues
#include "Prune.h"// for pruneLevelSet

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
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
///
/// @note The distance tree must hold @c float values (FloatTree or another
/// tree configuration with a float ValueType); this is checked at compile
/// time. The label tree may hold any integer type (Int32Tree by default;
/// Int64Tree and UInt32Tree also work).
template<typename FloatTreeT = FloatTree, typename LabelTreeT = Int32Tree>
class MultiLevelSet
{
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "MultiLevelSet: the distance tree must have float values");
    static_assert(std::is_integral<typename LabelTreeT::ValueType>::value,
        "MultiLevelSet: the label tree must have integer values");

public:
    using Ptr = SharedPtr<MultiLevelSet>;
    using FloatGridT = Grid<FloatTreeT>;
    using LabelGridT = Grid<LabelTreeT>;
    using LabelT = typename LabelTreeT::ValueType;

    /// @brief Pair up @a phi and @a labels.
    /// @throw ValueError if a grid is null, the transforms differ, the
    /// transform is not a uniform scale, @a halfWidth is less than 2, the
    /// phi background differs from <tt>halfWidth * voxelSize</tt> (relative
    /// tolerance 1e-6), the label background is not 0, or phi is negative or
    /// not finite at an active voxel.
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
            OPENVDB_THROW(ValueError,
                "MultiLevelSet: phi and label grids must share a transform");
        }
        if (!mPhi->transform().hasUniformScale()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: transform must have uniform voxel size");
        }
        std::string message;
        if (!this->validateBackgrounds(&message) || !this->validatePhi(&message)) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: " << message);
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
    /// face midpoint). They are kept frozen while the rest of the band
    /// (interface voxels dilated by @a halfWidth over 26-neighbours) is
    /// redistanced; the interface values are not re-estimated with a subcell
    /// fix, because that gradient estimate needs a distance field, not this
    /// voxel staircase. Voxels left with phi >= halfWidth*dx are deactivated.
    /// Every inactive voxel kept in the trees holds its label from @a labels,
    /// so the label of a region's interior is known just outside the band.
    /// @param grainSize TBB grain size in leaves; zero or negative runs
    /// serially. The result does not depend on it.
    /// @throw ValueError if @a halfWidth < 2, the voxel size is not uniform
    /// or the label background is not 0.
    static Ptr fromLabelGrid(const LabelGridT& labels, int halfWidth = 3, int grainSize = 1);

    /// @brief Build a multi-object level set from signed distance grids, one
    /// per non-zero label.
    /// @details psi_l is the input grid of label l (inactive values read
    /// with their sign, so deep interiors are negative); the background
    /// label 0 has the implicit psi_0 = -min_l psi_l, i.e. it is the region
    /// outside every object. At each voxel the label is argmin psi (ties to
    /// the smaller label) and phi = (v_b - v_a)/2 for the two smallest values
    /// v_a <= v_b, so phi = |psi_l| near an isolated object. Candidates are
    /// the union of the input active topologies and their face neighbours;
    /// the band is the interface voxels dilated by @a halfWidth
    /// (26-neighbours), then reinitialized (interface voxels re-estimated
    /// with the Russo-Smereka subcell fix, the rest of the band solved from
    /// them; see MultiLevelSetRedistance::reinitialize) and trimmed to
    /// phi < halfWidth*dx. Inactive tiles and voxels kept in the label tree
    /// hold the label of the region they lie in, so the label of a region's
    /// interior is known just outside the band. Inactive tiles are labelled
    /// by the argmin at the tile origin, which is exact for disjoint inputs;
    /// where inputs overlap inside a tile, only voxels in leaves (of the
    /// band, its face shell or an input) are labelled per voxel.
    /// @param grainSize TBB grain size in leaves; zero or negative runs
    /// serially. The result does not depend on it.
    /// @throw ValueError if the map is empty, a grid is null or not a
    /// GRID_LEVEL_SET, a key is 0, the transforms differ or are not a uniform
    /// scale, or @a halfWidth < 2.
    static Ptr fromSignedGrids(
        const std::map<int32_t, typename FloatGridT::ConstPtr>& grids, int halfWidth = 3,
        int grainSize = 1);

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

    /// @brief Signed distance psi to the boundary of the region of
    /// @a label: second-order accurate away from junction lines (where three
    /// or more labels meet), first-order within about two voxels of one,
    /// where the distance field itself is not smooth.
    /// @details A voxel is inside if its label (active or not; inactive
    /// voxels keep the label of the region they lie in) equals @a label.
    /// Seeds are the active phi voxels with a face neighbour on the other
    /// side of that region; they keep their phi, which is already the
    /// distance to the region's boundary (re-estimating it with a subcell
    /// fix would move the interface, since that fix is not a fixed point of
    /// a redistanced field). The exception is an outside seed next to a third
    /// label: its phi may be the distance to the boundary with that third
    /// label, so it is rebuilt from inside values within proven bounds (see
    /// lsmo_internal::rebuildJunctionSeeds). The other band voxels solve
    /// |grad psi| = 1 from the seeds as in MultiLevelSetRedistance, on |psi|,
    /// and the sign is applied last. The result is active exactly at the
    /// voxels with |psi| < halfWidth*dx (the seeds dilated by @a halfWidth;
    /// for @a halfWidth up to this object's halfWidth() the search is limited
    /// to the active band, which already contains every such voxel).
    /// Inactive values are -/+ halfWidth*dx by region.
    ///
    /// Accuracy: the layers next to the boundary are accurate to the order
    /// stated above in the maximum norm; the outer layers of the band are
    /// accurate in RMS but may have larger isolated errors, where the
    /// upwind solve meets characteristics coming from far-apart seeds.
    /// @param halfWidth band half-width in voxels of the result; -1 (the
    /// default) uses this object's halfWidth(); otherwise it must be at
    /// least 2. Values above halfWidth() are allowed, but voxels beyond this
    /// object's band have no phi data and are solved from the band only.
    /// @param grainSize TBB grain size in leaves; zero or negative runs
    /// serially. The result does not depend on it.
    /// @param interrupt optional interrupter, polled between Jacobi
    /// iterations; if it fires, the result is a valid level set grid but
    /// voxels not yet reached hold the background value.
    /// @return a GRID_LEVEL_SET grid with background halfWidth*dx and this
    /// object's transform; it has no active voxels if @a label has no
    /// boundary in the band (e.g. it does not occur).
    /// @throw ValueError if @a halfWidth is not -1 and less than 2.
    template<typename InterruptT = util::NullInterrupter>
    typename FloatGridT::Ptr extractSignedDistance(int32_t label, int halfWidth = -1,
        int grainSize = 1, InterruptT* interrupt = nullptr) const;

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

    /// @brief Check the class invariant (identical active topology, phi
    /// finite and non-negative at active voxels, phi background
    /// halfWidth*dx and label background 0).
    /// @param message if non-null, receives a description of the first failure.
    bool validate(std::string* message = nullptr) const
    {
        if (!this->validateBackgrounds(message)) return false;
        if (!mPhi->tree().hasSameTopology(mLabels->tree())) {
            if (message) *message = "phi and label trees have different active topology";
            return false;
        }
        return this->validatePhi(message);
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
        const BoolTreeT& candidates, const SamplerT& sampler, int halfWidth, bool reseed,
        int grainSize);

    /// phi background == halfWidth*dx (relative tolerance 1e-6), label background 0.
    bool validateBackgrounds(std::string* message) const
    {
        const double expected = double(mHalfWidth) * this->voxelSize();
        const double bg = double(mPhi->tree().background());
        if (!(std::abs(bg - expected) <= 1e-6 * expected)) {
            if (message) {
                std::ostringstream ss;
                ss << "phi background " << bg << " differs from halfWidth*voxelSize = "
                   << expected;
                *message = ss.str();
            }
            return false;
        }
        if (mLabels->tree().background() != LabelT(0)) {
            if (message) *message = "label background is not 0";
            return false;
        }
        return true;
    }

    /// phi finite and non-negative at every active voxel.
    bool validatePhi(std::string* message) const
    {
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

/// Run @a body on the leaf range of @a leafs: in parallel with grain size
/// @a grainSize, or serially if @a grainSize is zero or negative.
template<typename LeafManagerT, typename BodyT>
void forEachLeafRange(LeafManagerT& leafs, int grainSize, const BodyT& body)
{
    if (grainSize > 0) tbb::parallel_for(leafs.leafRange(size_t(grainSize)), body);
    else body(leafs.leafRange());
}

/// @brief Interface seeding over the active voxels of @a out.
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
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
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

/// @brief Outer-layer solver on the unsigned distance @a phi.
/// @details Active voxels of @a phi that are not active in @a seeds are reset
/// and solve |grad phi| = 1 by parallel Jacobi iteration with the seeds as
/// Dirichlet data: second-order one-sided upwind differences (Chopp 2001)
/// where two same-side upwind values are available (the outer one, q2, with
/// !differs(p, q2) and strictly smaller than the inner one) and first order
/// otherwise. Inactive neighbours are not used. Iteration stops when
/// max|delta phi| < @a tolerance, after @a maxIter iterations, or on
/// interrupt; voxels left unreached get the background of @a phi.
/// With @a signedReflection (used by the evolver, which freezes only the
/// voxels it just projected and must re-solve their other neighbours on both
/// sides of a moving interface) a face neighbour q of p
/// contributes +phi_q if !differs(p, q), -phi_q if differs(p, q) and q is a
/// seed, and nothing otherwise; the second-order term then also needs q on
/// p's side, and results are clamped to >= 0. This is exact for planar
/// interfaces whose frozen voxels lie on either side.
/// @param differs copied once per task; called as differs(p, q).
/// @param grainSize TBB grain size in leaves; zero or negative runs serially.
/// @return the number of Jacobi iterations performed.
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp, typename InterruptT>
int solveOuterLayers(FloatTreeT& phi, const BoolTreeT& seeds, const DiffersOp& differs,
    double dx, float tolerance, int maxIter, int grainSize, InterruptT* interrupt,
    bool signedReflection = false)
{
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
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
            tree::ValueAccessor<const BoolTreeT> seedAcc(seeds);
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
                            if (!acc.probeValue(q, v)) continue;
                            bool sameSide = true;
                            if (signedReflection && op(p, q)) {
                                if (!seedAcc.isValueOn(q)) continue; // unknown across
                                v = -v; // frozen voxel across the interface
                                sameSide = false;
                            }
                            if (!(v < a)) continue;
                            a = v;
                            // a' must lie on p's side of the interface
                            const Coord q2 = q + offs[2 * axis + side];
                            if (!sameSide || !acc.probeValue(q2, a2) || op(p, q2)) {
                                a2 = kUnknownDistance;
                            }
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
                    float upd = float(solveGodunov(abar, coef, n, dx));
                    if (signedReflection) upd = std::max(upd, 0.0f);
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


/// @brief Second-order label-aware redistancing of a MultiLevelSet.
///
/// @details Restores phi = distance to the voxel's own label boundary over
/// the active band while keeping the interface fixed:
/// -# Interface voxels (an active voxel with a face neighbour, active or
///    not, of a different label) are frozen. redistance() keeps their
///    current values, so repeated calls never move a crossing (use it every
///    evolution step). reinitialize() first reseeds them with the
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
/// inactive face neighbours of the band (inactive label voxels keep the
/// label of the region they lie in, see MultiLevelSet). A band-edge voxel
/// whose inactive neighbour reads the label background instead becomes a
/// spurious interface voxel.
///
/// @note FastSweeping.h was evaluated for reuse and rejected: its
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

    /// @brief Reseed the interface voxels with the Russo-Smereka subcell distance,
    /// then redistance phi in place.
    /// @return the number of Jacobi iterations performed.
    int reinitialize() { return this->run(/*reseed=*/true); }

    /// @brief Redistance with only the active voxels of @a freeze kept
    /// (used by the evolver after each step, with the voxels it projected
    /// frozen). Every other active voxel solves |grad phi| = 1 as above,
    /// with signed reflection: a face neighbour of the same label gives
    /// +phi, a frozen face neighbour of a different label gives -phi, other
    /// neighbours of a different label are ignored. Unlike redistance(),
    /// voxels next to a label change that are not in @a freeze are solved,
    /// not frozen.
    /// @return the number of Jacobi iterations performed.
    int redistance(const BoolTreeT& freeze)
    {
        FloatTreeT& phi = mMls.phi()->tree();
        const double dx = mMls.voxelSize();
        const int maxIter = int(std::ceil(mMls.halfWidth() * std::sqrt(3.0))) + 2;
        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");
        BoolTreeT seeds(freeze);
        seeds.voxelizeActiveTiles();
        seeds.topologyIntersection(phi);
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        const int iter = lsmo_internal::solveOuterLayers(phi, seeds, differs, dx,
            mTolerance, maxIter, mGrainSize, mInterrupter, /*signedReflection=*/true);
        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    /// @brief Interface seeding only, for per-label reconstruction.
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

        // Interface voxels: frozen as they are, or reseeded (subcell fix)
        // from a read-only snapshot.
        BoolTreeT seeds(false);
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        if (reseed) {
            const FloatTreeT snapshot(phi);
            lsmo_internal::seedInterface(snapshot, differs, phi, seeds, dx, mGrainSize);
        } else {
            lsmo_internal::seedInterface(phi, differs, phi, seeds, dx, mGrainSize,
                /*writeValues=*/false);
        }

        // outer layers
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

/// @brief Per-label reconstruction of seeds near junctions, for
/// MultiLevelSet::extractSignedDistance.
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
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
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
                    ball = std::min(ball,
                        dx * std::sqrt(double(i * i + j * j + k * k)) - double(v));
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
/// background (the region outside every object); the label is the smallest
/// candidate (ties to the smaller label) and phi = (v_b - v_a)/2 for the two
/// smallest v_a <= v_b.
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
///    phi; then reinitialize() (subcell reseed of the interface voxels, then
///    the outer-layer solve) if @a reseed, else redistance() (sampled
///    interface values frozen, then the outer-layer solve).
/// -# Voxels with phi >= halfWidth*dx are deactivated (labels kept), then
///    pruneInactive().
template<typename FloatTreeT, typename LabelTreeT>
template<typename SamplerT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::build(const math::Transform& xform,
    typename LabelTreeT::Ptr labels, const BoolTreeT& candidates,
    const SamplerT& sampler, int halfWidth, bool reseed, int grainSize)
{
    const double dx = xform.voxelSize()[0];
    const float bg = float(halfWidth * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // 1. interface voxels
    BoolTreeT band(candidates);
    band.voxelizeActiveTiles();
    {
        tree::LeafManager<BoolTreeT> leafs(band);
        lsmo_internal::forEachLeafRange(leafs, grainSize,
            [&](const typename tree::LeafManager<BoolTreeT>::LeafRange& range) {
                SamplerT s(sampler);
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    for (auto it = leaf->beginValueOn(); it; ++it) {
                        const Coord p = it.getCoord();
                        const int32_t l = s.label(p);
                        bool iface = false;
                        for (int f = 0; f < 6 && !iface; ++f) {
                            iface = s.label(p + offs[f]) != l;
                        }
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
        lsmo_internal::forEachLeafRange(leafs, grainSize,
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
        lsmo_internal::forEachLeafRange(leafs, grainSize,
            [&](const typename tree::LeafManager<FloatTreeT>::LeafRange& range) {
                SamplerT s(sampler);
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    for (auto it = leaf->beginValueOn(); it; ++it) {
                        it.setValue(s.phi(it.getCoord()));
                    }
                }
            });
    }
    typename FloatGridT::Ptr phiGrid = FloatGridT::create(phiTree);
    phiGrid->setTransform(xform.copy());
    typename LabelGridT::Ptr labGrid = LabelGridT::create(labels);
    labGrid->setTransform(xform.copy());
    Ptr mls(new MultiLevelSet(phiGrid, labGrid, halfWidth));
    MultiLevelSetRedistance<MultiLevelSet> redist(*mls);
    redist.setGrainSize(grainSize);
    if (reseed) redist.reinitialize(); else redist.redistance();

    // 4. trim to the band phi < halfWidth*dx
    {
        tree::LeafManager<FloatTreeT> leafs(*phiTree);
        lsmo_internal::forEachLeafRange(leafs, grainSize,
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
template<typename InterruptT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::FloatGridT::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::extractSignedDistance(int32_t label, int halfWidth,
    int grainSize, InterruptT* interrupt) const
{
    if (halfWidth == -1) halfWidth = mHalfWidth;
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError,
            "MultiLevelSet::extractSignedDistance: halfWidth must be at least 2");
    }
    using RegionDiffers = lsmo_internal::LabelRegionDiffers<FloatTreeT, LabelTreeT>;
    using BoolLeafRangeT = typename tree::LeafManager<BoolTreeT>::LeafRange;
    using LeafRangeT = typename tree::LeafManager<FloatTreeT>::LeafRange;
    const FloatTreeT& phi = mPhi->tree();
    const LabelTreeT& labels = mLabels->tree();
    const double dx = this->voxelSize();
    const float bg = float(halfWidth * dx);
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
        lsmo_internal::forEachLeafRange(leafs, grainSize, [&](const BoolLeafRangeT& range) {
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
        lsmo_internal::forEachLeafRange(leafs, grainSize, [&](const LeafRangeT& range) {
            tree::ValueAccessor<const FloatTreeT> acc(phi);
            for (auto leaf = range.begin(); leaf; ++leaf) {
                for (auto it = leaf->beginValueOn(); it; ++it) {
                    it.setValue(acc.getValue(it.getCoord()));
                }
            }
        });
    }

    // 3. seeds: label | not-label crossings, keeping the pair's phi (no
    // subcell reseed: it is not a fixed point of a redistanced phi, so it
    // would move the interface). Not-label seeds next to a third label may
    // store the distance to the wrong boundary and are rebuilt from
    // label-side data.
    if (interrupt) interrupt->start("Extracting signed distance");
    BoolTreeT seeds(false);
    lsmo_internal::seedInterface(phi, seedOp, *out, seeds, dx, grainSize,
        /*writeValues=*/false);
    lsmo_internal::rebuildJunctionSeeds(*out, seeds, labels, inLabel, dx);
    const int maxIter = int(std::ceil(halfWidth * std::sqrt(3.0))) + 2;
    lsmo_internal::solveOuterLayers(*out, seeds, sideOp, dx, float(1e-4 * dx), maxIter,
        grainSize, interrupt);
    if (interrupt) interrupt->end();

    // 4. sign by region, trim to |psi| < halfWidth*dx, inactive values -/+bg
    {
        tree::LeafManager<FloatTreeT> leafs(*out);
        lsmo_internal::forEachLeafRange(leafs, grainSize, [&](const LeafRangeT& range) {
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
    tools::pruneLevelSet(*out, /*threaded=*/grainSize > 0, size_t(std::max(grainSize, 1)));

    typename FloatGridT::Ptr grid = FloatGridT::create(out);
    grid->setTransform(mPhi->transform().copy());
    grid->setGridClass(GRID_LEVEL_SET);
    return grid;
}


template<typename FloatTreeT, typename LabelTreeT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::fromLabelGrid(const LabelGridT& labels, int halfWidth,
    int grainSize)
{
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: halfWidth must be at least 2");
    }
    if (!labels.transform().hasUniformScale()) {
        OPENVDB_THROW(ValueError,
            "MultiLevelSet::fromLabelGrid: transform must have uniform voxel size");
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
        halfWidth, /*reseed=*/false, grainSize);
}


template<typename FloatTreeT, typename LabelTreeT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::fromSignedGrids(
    const std::map<int32_t, typename FloatGridT::ConstPtr>& grids, int halfWidth, int grainSize)
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
            OPENVDB_THROW(ValueError,
                "MultiLevelSet::fromSignedGrids: label 0 is reserved for the background");
        }
        if (kv.second->getGridClass() != GRID_LEVEL_SET) {
            OPENVDB_THROW(ValueError,
                "MultiLevelSet::fromSignedGrids: input grids must be level sets");
        }
        if (!xform) {
            xform = &kv.second->transform();
        } else if (*xform != kv.second->transform()) {
            OPENVDB_THROW(ValueError,
                "MultiLevelSet::fromSignedGrids: input grids must share a transform");
        }
        inputs.emplace_back(kv.first, &kv.second->tree());
    }
    if (!xform->hasUniformScale()) {
        OPENVDB_THROW(ValueError,
            "MultiLevelSet::fromSignedGrids: transform must have uniform voxel size");
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

    return build(*xform, lab, candidates, sampler, halfWidth, /*reseed=*/true, grainSize);
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
///
/// The stencil also samples auxiliary grids (velocity, pressure, ...) at its
/// centre voxel through sampleAtCentre(), which caches the result per centre
/// voxel so that speed functors evaluating several labels at one voxel sample
/// each field once.
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
        if (ijk != mCenter) {
            mScalarCache.valid = false;
            mVectorCache.valid = false;
        }
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

    /// @brief BoxSampler value of @a grid at the world position of the
    /// centre voxel. The value is cached until the stencil moves to another
    /// voxel (or another grid is sampled), and the grid is read through a
    /// value accessor kept by the stencil. @a grid must outlive the stencil.
    float sampleAtCentre(const FloatGrid& grid) const
    {
        return this->sample(grid, mScalarCache);
    }
    /// @brief Vector-valued version of sampleAtCentre().
    Vec3f sampleAtCentre(const Vec3SGrid& grid) const
    {
        return this->sample(grid, mVectorCache);
    }

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
    /// One-entry cache of a sampled grid: the grid, a value accessor to its
    /// tree, and the value at the current centre voxel.
    template<typename GridT>
    struct FieldCache
    {
        const GridT* grid = nullptr;
        std::unique_ptr<tree::ValueAccessor<const typename GridT::TreeType>> acc;
        bool valid = false;
        typename GridT::ValueType value{};
    };

    template<typename GridT>
    typename GridT::ValueType sample(const GridT& grid, FieldCache<GridT>& cache) const
    {
        if (cache.grid != &grid) {
            cache.grid = &grid;
            cache.acc.reset(new tree::ValueAccessor<const typename GridT::TreeType>(grid.tree()));
            cache.valid = false;
        }
        if (!cache.valid) {
            const Vec3d w = mTransform.indexToWorld(mCenter);
            const Vec3R idx = grid.transform().worldToIndex(w);
            cache.value = tools::BoxSampler::sample(*cache.acc, idx);
            cache.valid = true;
        }
        return cache.value;
    }

    const math::Transform& mTransform;
    tree::ValueAccessor<const FloatTreeT> mPhiAcc;
    tree::ValueAccessor<const LabelTreeT> mLabelAcc;
    float mDx;
    Coord mCenter;
    int32_t mLabel = 0;
    float mValues[19];
    mutable FieldCache<FloatGrid> mScalarCache;
    mutable FieldCache<Vec3SGrid> mVectorCache;
};


/// @brief Built-in speed functor: the MOGAC multi-object force model of
/// curvature + upwind pressure + upwind advection, as in the reference
/// implementation this port follows.
/// @details For label l at a voxel it returns dpsi_l/dt =
/// w_c*kappa*|grad psi| (clamped to the maximum curvature force)
/// - (upwind advection v . grad psi) - (upwind pressure F*|grad psi|).
/// It satisfies the SpeedT concept of LevelSetMultiObject. Label 0 is
/// handled by the evolver as passive (its value is the complement of the
/// other candidates, so it never needs a speed), so isFixed(0) is false
/// here. Speeds are world-space; with unit voxel size they equal the
/// reference implementation's index-space values. operator() is const and
/// thread-safe.
/// @note Unlike the reference implementation, the curvature term uses the
/// curvature of the interface, k_x/(1 - psi*k_x), rather than the mean
/// curvature k_x of the level set through the voxel centre (falling back to
/// k_x when |psi*k_x| >= 0.5), which makes the evaluation point exact for a
/// sphere. It does not remove the measured bias of mean curvature flow (a
/// sphere shrinks about 3.5% too fast; the correction changes r^2 at t = 20
/// only from 58.32 to 58.53 against the exact 60): that residual comes from
/// an error of about 0.01*dx in the interface-voxel phi, amplified by the
/// second differences, and is comparable to that of a single-object
/// LevelSetAdvection reference on the same problem.
template<typename MultiLevelSetT>
class MultiObjectForces
{
public:
    /// Curvature weight w: speed w*kappa*|grad psi| (clamped), with kappa the
    /// interface mean curvature k_x/(1 - psi*k_x) (see the class note).
    void setCurvatureWeight(float w) { mCurvatureWeight = w; }
    void setMaxCurvatureForce(float m = 10.0f) { mMaxCurvatureForce = m; }

    /// @brief Pressure force field. A deep copy is rescaled once, as in the
    /// reference implementation: values are shifted by @a target and the negative and
    /// positive parts scaled independently to [-1, 1] (scale 1 for a part
    /// whose extreme magnitude is at most 1e-4). A NaN @a target (the
    /// default) skips the rescale.
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
        OPENVDB_ASSERT(ijk == s.center());
        (void)ijk; // fields are sampled at s.center()
        if (!isActive(label)) return 0.999f;
        if (mFixed.count(label)) return 0.0f;

        float dneg[3], dpos[3];
        s.upwindDifferences(dneg, dpos);

        float kappa = 0.0f;
        if (mCurvatureWeight != 0.0f) {
            // Interface curvature: the voxel's level set has mean curvature
            // k_x; the interface, at signed distance -psi along the normal,
            // has k = k_x / (1 - psi*k_x) for a sphere (a first-order
            // correction otherwise). Falls back to k_x when |psi*k_x| >= 0.5.
            float kn = s.meanCurvatureNormGrad(); // k_x * |grad psi|
            const float g = s.gradient().length();
            if (g > 0.0f) {
                const float pk = s.template getValue<0, 0, 0>() * kn / g; // psi*k_x
                if (std::abs(pk) < 0.5f) kn /= (1.0f - pk);
            }
            kappa = mCurvatureWeight * kn;
            kappa = std::max(-mMaxCurvatureForce, std::min(mMaxCurvatureForce, kappa));
        }

        // fields are sampled at the stencil centre (== ijk), once per voxel
        float advection = 0.0f;
        if (mAdvection) {
            const Vec3f vel = s.sampleAtCentre(*mAdvection);
            for (int a = 0; a < 3; ++a) {
                const float f = mAdvectionWeight * vel[a];
                if (f > 0.0f) advection += f * dneg[a];
                else if (f < 0.0f) advection += f * dpos[a];
            }
        }

        float pressure = 0.0f;
        if (mPressure) {
            const float force = mPressureWeight * s.sampleAtCentre(*mPressure);
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
    /// evolver treats it as passive and never asks the functor for its speed.
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


/// @brief Digital adjacency pair (foreground, background) used by
/// topology-preserving evolution and the multi-label mesher.
enum class MultiObjectConnectivity { FG26_BG6, FG6_BG26, FG18_BG6, FG6_BG18 };


namespace lsmo_internal {

/// Adjacency tables of the 3x3x3 neighbourhood, bit i <-> (dx,dy,dz) with
/// i = (dx+1) + 3(dy+1) + 9(dz+1): adj[a][i] holds the positions other than
/// the centre that are a-adjacent to i (a = 0: 6, 1: 18, 2: 26), and
/// star[a] the a-neighbours of the centre (N*_6, N*_18, N*_26).
struct SimplePointTables
{
    uint32_t adj[3][27];
    uint32_t star[3];
    SimplePointTables()
    {
        auto coord = [](int i) { return Coord(i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1); };
        const uint32_t centre = uint32_t(1) << 13;
        for (int a = 0; a < 3; ++a) {
            star[a] = 0;
            for (int i = 0; i < 27; ++i) {
                adj[a][i] = 0;
                for (int j = 0; j < 27; ++j) {
                    const Coord d = coord(i) - coord(j);
                    const int d2 = d.x() * d.x() + d.y() * d.y() + d.z() * d.z();
                    if (d2 >= 1 && d2 <= a + 1) adj[a][i] |= uint32_t(1) << j;
                }
                adj[a][i] &= ~centre;
            }
            star[a] = adj[a][13];
        }
    }
};

inline const SimplePointTables& simplePointTables()
{
    static const SimplePointTables tables;
    return tables;
}

/// Number of components under adjacency @a a of the positions in @a set.
inline int countComponents(uint32_t set, int a)
{
    const SimplePointTables& t = simplePointTables();
    int count = 0;
    while (set) {
        uint32_t comp = set & (~set + 1); // lowest bit
        uint32_t frontier = comp;
        while (frontier) {
            const int i = int(util::FindLowestOn(Index32(frontier)));
            frontier &= frontier - 1;
            const uint32_t add = t.adj[a][i] & set & ~comp;
            comp |= add;
            frontier |= add;
        }
        set &= ~comp;
        ++count;
    }
    return count;
}

/// Bertrand-Malandain topological number T_n(x, X) of the centre x for the
/// positions @a X of the 26-neighbourhood (centre excluded): the number of
/// n-components of the geodesic neighbourhood G_n(x, X) = N^k_n(x, X), with
/// N^1_n = N*_n(x) & X and N^k_n = union of N_n(y) & N*_26(x) & X over y in
/// N^{k-1}_n; k = 2 for 6 (paired with 26) and 18, k = 3 for 6+ (6 paired
/// with 18), k = 1 for 26. @a a is 0, 1, 2 for n = 6, 18, 26.
inline int topologicalNumber(uint32_t X, int a, int k)
{
    const SimplePointTables& t = simplePointTables();
    X &= t.star[2];
    uint32_t g = X & t.star[a];
    for (int level = 2; level <= k; ++level) {
        uint32_t next = g;
        for (uint32_t b = g; b; b &= b - 1) {
            next |= t.adj[a][int(util::FindLowestOn(Index32(b)))] & X;
        }
        g = next;
    }
    return countComponents(g, a);
}

/// @brief Whether the centre of a 3x3x3 neighbourhood is a simple point of
/// the foreground region @a inMask (27 bits, bit i <-> (dx,dy,dz) with
/// i = (dx+1) + 3(dy+1) + 9(dz+1); the centre bit is ignored) under the
/// adjacency pair @a conn: adding or removing it changes neither the
/// region's nor its complement's topology (no new or lost components,
/// tunnels or cavities). Decided by T_fg = 1 and T_bg = 1 (Bertrand and
/// Malandain 1994).
inline bool isSimplePoint(uint32_t inMask, MultiObjectConnectivity conn)
{
    const uint32_t all = (uint32_t(1) << 27) - 1, centre = uint32_t(1) << 13;
    const uint32_t fg = inMask & all & ~centre, bg = ~inMask & all & ~centre;
    switch (conn) {
        case MultiObjectConnectivity::FG26_BG6:
            return topologicalNumber(fg, 2, 1) == 1 && topologicalNumber(bg, 0, 2) == 1;
        case MultiObjectConnectivity::FG6_BG26:
            return topologicalNumber(fg, 0, 2) == 1 && topologicalNumber(bg, 2, 1) == 1;
        case MultiObjectConnectivity::FG18_BG6:
            return topologicalNumber(fg, 1, 2) == 1 && topologicalNumber(bg, 0, 3) == 1;
        case MultiObjectConnectivity::FG6_BG18:
            return topologicalNumber(fg, 0, 3) == 1 && topologicalNumber(bg, 1, 2) == 1;
    }
    return false;
}

} // namespace lsmo_internal


/// @brief Multi-object level set evolver (multi-object geodesic active
/// contours, MOGAC).
///
/// @details One forward Euler step of step() does:
/// -# <b>Projected set and candidates.</b> Every active voxel that is not of
///    a fixed label and has phi <= 0.5*dx or a face neighbour (active or
///    not) of a different label is projected: these are the only voxels
///    whose label can change in one step (see Time step), so the update is
///    restricted to them. Its candidates are its own label and the labels
///    of its face neighbours, minus fixed labels. Label 0 is passive: it is a
///    candidate only if it is the voxel's or a face neighbour's label or the
///    voxel touches a fixed label, and its value is the complement
///    v_0 = -min v_l over the non-zero candidates, so object/background
///    interfaces move at the object's full speed (and a voxel next to a
///    retreating fixed label falls back to label 0). The speed functor is
///    called (on time-n data through a LabelAwareStencil) only for non-zero
///    candidates; labels with isActive() false get dpsi/dt = +0.999, so
///    they slowly retreat from their neighbours.
/// -# <b>Time step.</b> dt = min(maxDt, 0.5*cfl*dx/max|F|, speed.stableDt(dx)),
///    so an interface moves at most half a voxel per step; with
///    setClampSpeed(true), F is clamped to [-1, 1] and
///    dt = min(maxDt, 0.5*dx, speed.stableDt(dx)). The half-voxel cap is
///    what makes the projected set sufficient: since psi_l >= phi at every
///    voxel, no voxel outside it can change label by projection (orphan
///    cleanup can still relabel), so no separate propagation pass is needed.
/// -# <b>Projection.</b> v_l = psi_l + dt*F_l with psi_l = -phi for the own
///    label and +phi otherwise; the new label is the smallest v (ties to the
///    smaller label) and phi = |v_1 - v_2|/2 over the two smallest values, or
///    |v_1| with one candidate. Results are written to copies of both trees
///    (time-n data stays readable until both are swapped in together).
/// -# <b>Topology preservation</b> (opt-in): with
///    setTopologyPreservingLabels(), a projected change a -> b that involves
///    a protected label is deferred and applied in 27 coloured passes
///    (coordinates mod 3) only if the voxel is a simple point
///    (lsmo_internal::isSimplePoint, under setConnectivity()) of every
///    protected label in {a, b} on the labels as updated by earlier passes;
///    a rejected voxel keeps label a with phi = 1e-3*dx and stays frozen.
///    Orphan cleanup and band activation never change a protected label's
///    region except by simple changes, so its components, cavities and
///    tunnels are preserved. With no protected label nothing changes.
/// -# <b>Orphan cleanup</b>: a voxel whose label occurs nowhere in its
///    26-neighbourhood takes the label and phi of its smallest-phi active
///    neighbour (fixed labels excluded) and is redistanced (not frozen).
/// -# <b>Band maintenance</b>: both trees are swapped in; inactive face
///    neighbours of active voxels with phi < (halfWidth-1)*dx are activated
///    (phi = background, label of the smallest-phi active face neighbour);
///    MultiLevelSetRedistance::redistance(freeze) keeps the projected voxels
///    and the fixed-label voxels and solves all others with signed
///    reflection, so voxels the interface just passed are re-solved from
///    both sides; voxels with phi >= halfWidth*dx are deactivated in
///    both trees (labels kept, so the interior label stays known);
///    pruneInactive(). Builds with OPENVDB_ENABLE_ASSERTS check validate()
///    after every step.
///
/// Voxels of fixed labels never change label or phi and no label can
/// capture them; a step with no non-zero candidate anywhere changes nothing.
///
/// @par SpeedT concept
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
    using BoolTreeT = typename FloatTreeT::template ValueConverter<bool>::Type;
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
    /// @brief Clamp speeds to [-1, 1] and take dt = min(maxDt, 0.5*dx)
    /// (capped by speed.stableDt), as the reference implementation's
    /// clampSpeed option.
    void setClampSpeed(bool on) { mClampSpeed = on; }
    /// @brief TBB grain size in leaves; zero or negative runs serially.
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }
    /// @brief Redistance tolerance in world units (default 1e-4*dx).
    void setRedistanceTolerance(float tolWorld) { mTolerance = tolWorld; }
    /// @brief Labels whose digital topology is preserved (default
    /// none): a label change a -> b at a voxel is applied only if the voxel is
    /// a simple point of every protected label in {a, b}.
    void setTopologyPreservingLabels(std::set<int32_t> labels) { mProtected = std::move(labels); }
    /// @brief Adjacency pair of topology preservation (default FG26_BG6).
    void setConnectivity(MultiObjectConnectivity conn) { mConnectivity = conn; }

    float cfl() const { return mCfl; }
    bool clampSpeed() const { return mClampSpeed; }
    int grainSize() const { return mGrainSize; }
    float redistanceTolerance() const { return mTolerance; }
    const std::set<int32_t>& topologyPreservingLabels() const { return mProtected; }
    MultiObjectConnectivity connectivity() const { return mConnectivity; }

    /// @brief One forward Euler step of at most @a maxDt (world time).
    /// @return the time step taken; min(@a maxDt, speed.stableDt) without
    /// changes if nothing can move, 0 if @a maxDt <= 0.
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
    /// Own label + 6 face labels.
    static constexpr int kMaxCandidates = 7;

    /// Candidate list of one projected voxel: non-zero labels with their
    /// speeds, and whether passive label 0 participates.
    struct VoxelEntry
    {
        Index offset = 0;
        int32_t own = 0;
        float phi = 0.0f; // time n
        int n = 0;
        bool zero = false;
        std::array<std::pair<int32_t, float>, kMaxCandidates> cand;
    };

    struct LeafData
    {
        std::vector<VoxelEntry> entries;
        float maxSpeed = 0.0f;
        bool any = false;
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

    bool isFixed(int32_t l) const { return l != 0 && mSpeed.isFixed(l); }

    bool isProtected(int32_t l) const { return !mProtected.empty() && mProtected.count(l) != 0; }
    /// Whether a change @a from -> @a to involves a protected label.
    bool involvesProtected(int32_t from, int32_t to) const
    {
        return from != to && (this->isProtected(from) || this->isProtected(to));
    }

    /// Whether changing the label of @a p from @a from to @a to keeps every
    /// protected label among them topologically unchanged, on @a labels.
    template<typename AccT>
    bool topologyAllows(AccT& labels, const Coord& p, int32_t from, int32_t to) const
    {
        for (const int32_t l : {from, to}) {
            if (!this->isProtected(l)) continue;
            uint32_t mask = 0;
            for (int i = 0; i < 27; ++i) {
                const Coord q = p.offsetBy(i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1);
                if (static_cast<int32_t>(labels.getValue(q)) == l) mask |= uint32_t(1) << i;
            }
            if (!lsmo_internal::isSimplePoint(mask, mConnectivity)) return false;
        }
        return true;
    }

    /// A projected label change involving a protected label.
    struct Deferred { Coord ijk; int32_t from, to; float phi; };
    void applyDeferred(const std::vector<std::vector<Deferred>>& deferred,
        FloatTreeT& newPhi, LabelTreeT& newLabels, BoolTreeT& freeze);

    /// Add @a l to @a e's candidates: label 0 and fixed labels only make
    /// passive label 0 participate (a fixed label can never be captured).
    void addCandidate(VoxelEntry& e, int32_t l) const
    {
        if (l == 0 || mSpeed.isFixed(l)) { e.zero = true; return; } // passive 0 / wall
        for (int i = 0; i < e.n; ++i) if (e.cand[i].first == l) return;
        e.cand[e.n++] = std::make_pair(l, 0.0f); // at most 7 distinct labels
    }

    float evalSpeed(StencilT& s, const Coord& ijk, int32_t label) const
    {
        float f;
        if (!mSpeed.isActive(label)) {
            f = 0.999f; // inactive labels retreat
        } else {
            s.moveTo(ijk, label);
            f = mSpeed(ijk, label, s);
        }
        if (mClampSpeed) f = std::max(-1.0f, std::min(1.0f, f));
        return f;
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
        if (e.zero) consider(0, -minV); // passive label 0: complement
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

    void cleanOrphans(FloatTreeT& newPhi, LabelTreeT& newLabels, BoolTreeT& freeze);
    void bandMaintenance(const BoolTreeT& freeze);

    MultiLevelSetT& mMls;
    const SpeedT& mSpeed;
    InterruptT* mInterrupter;
    std::set<int32_t> mProtected;
    MultiObjectConnectivity mConnectivity = MultiObjectConnectivity::FG26_BG6;
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

    // Freeze mask: projected voxels and fixed-label voxels.
    BoolTreeT freeze(phi, false, TopologyCopy());
    freeze.voxelizeActiveTiles();
    for (auto leaf = freeze.beginLeaf(); leaf; ++leaf) leaf->setValuesOff();

    // 1-2. projected set, candidates and speeds, all on time-n data
    tree::LeafManager<const FloatTreeT> leafs(phi);
    const size_t leafCount = leafs.leafCount();
    std::vector<LeafData> data(leafCount);
    forEachIndex(leafCount, [&](size_t i) {
        const auto& leaf = leafs.leaf(i);
        LeafData& d = data[i];
        auto* maskLeaf = freeze.probeLeaf(leaf.origin());
        StencilT stencil(mMls);
        tree::ValueAccessor<const LabelTreeT> lab(labels);
        for (auto it = leaf.cbeginValueOn(); it; ++it) {
            const Coord p = it.getCoord();
            const int32_t own = static_cast<int32_t>(lab.getValue(p));
            if (this->isFixed(own)) { // walls are skipped and stay frozen
                if (maskLeaf) maskLeaf->setValueOn(it.pos());
                continue;
            }
            int32_t nb[6];
            bool seed = false;
            for (int f = 0; f < 6; ++f) {
                nb[f] = static_cast<int32_t>(lab.getValue(p + offs[f]));
                seed |= nb[f] != own;
            }
            if (!seed && *it > halfDx) continue;
            if (maskLeaf) maskLeaf->setValueOn(it.pos());
            d.entries.emplace_back();
            VoxelEntry& e = d.entries.back();
            e.offset = it.pos();
            e.own = own;
            e.phi = *it;
            this->addCandidate(e, own);
            for (int f = 0; f < 6; ++f) this->addCandidate(e, nb[f]);
            for (int c = 0; c < e.n; ++c) {
                const float s = this->evalSpeed(stencil, p, e.cand[c].first);
                e.cand[c].second = s;
                d.maxSpeed = std::max(d.maxSpeed, std::abs(s));
            }
            d.any |= e.n > 0;
        }
    });

    float maxSpeed = 0.0f;
    bool any = false;
    for (const LeafData& d : data) { maxSpeed = std::max(maxSpeed, d.maxSpeed); any |= d.any; }

    // 3. time step: interfaces move at most half a voxel, so only projected
    // voxels can change label
    float dt = std::min(maxDt, mSpeed.stableDt(float(dx)));
    if (!any) return dt; // no non-zero candidate anywhere: nothing moves
    if (mClampSpeed) dt = std::min(dt, halfDx);
    else if (maxSpeed > 0.0f) dt = std::min(dt, float(0.5 * mCfl * dx / maxSpeed));

    // 4. two-smallest projection, written to copies of both trees
    typename FloatTreeT::Ptr newPhi(new FloatTreeT(phi));
    typename LabelTreeT::Ptr newLabels(new LabelTreeT(labels));
    // changes involving a protected label are deferred (topology preservation)
    std::vector<std::vector<Deferred>> deferred(mProtected.empty() ? 0 : leafCount);
    forEachIndex(leafCount, [&](size_t i) {
        const LeafData& d = data[i];
        if (d.entries.empty()) return;
        const auto& leaf = leafs.leaf(i);
        auto* phiLeaf = newPhi->probeLeaf(leaf.origin());
        auto* labLeaf = newLabels->probeLeaf(leaf.origin());
        for (const VoxelEntry& e : d.entries) {
            int32_t l; float v;
            if (!project(e, dt, l, v)) continue;
            if (this->involvesProtected(e.own, l)) {
                deferred[i].push_back(Deferred{leaf.offsetToGlobalCoord(e.offset), e.own, l, v});
                continue;
            }
            phiLeaf->setValueOnly(e.offset, v);
            labLeaf->setValueOnly(e.offset, LabelT(l));
        }
    });
    if (!deferred.empty()) this->applyDeferred(deferred, *newPhi, *newLabels, freeze);

    // 5. orphan cleanup
    this->cleanOrphans(*newPhi, *newLabels, freeze);

    // 6a. swap both trees together; b-e. band maintenance
    this->swapBuffers(newPhi, newLabels);
    this->bandMaintenance(freeze);
    OPENVDB_ASSERT(mMls.validate());
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::cleanOrphans(
    FloatTreeT& newPhi, LabelTreeT& newLabels, BoolTreeT& freeze)
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
            if (this->isFixed(static_cast<int32_t>(own))) continue;
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
                        if (this->isFixed(static_cast<int32_t>(l))) continue;
                        if (v < best || (v == best && l < bestLabel)) { best = v; bestLabel = l; }
                    }
                }
            }
            if (!found && bestLabel != own) fixes[i].push_back(Fix{p, bestLabel, best});
        }
    });
    tree::ValueAccessor<FloatTreeT> phiAcc(newPhi);
    tree::ValueAccessor<LabelTreeT> labAcc(newLabels);
    tree::ValueAccessor<BoolTreeT> maskAcc(freeze);
    for (const auto& list : fixes) {
        for (const Fix& fx : list) {
            // protected labels: only simple changes, tested on the
            // labels as updated by the fixes applied so far (serial order)
            const int32_t own = static_cast<int32_t>(labAcc.getValue(fx.ijk));
            const int32_t to = static_cast<int32_t>(fx.label);
            if (this->involvesProtected(own, to)
                && !this->topologyAllows(labAcc, fx.ijk, own, to)) continue;
            phiAcc.setValueOnly(fx.ijk, fx.phi);
            labAcc.setValueOnly(fx.ijk, fx.label);
            maskAcc.setValueOff(fx.ijk); // flagged for redistancing
        }
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::applyDeferred(
    const std::vector<std::vector<Deferred>>& deferred,
    FloatTreeT& newPhi, LabelTreeT& newLabels, BoolTreeT& freeze)
{
    // 27 coloured passes (coordinates mod 3): changes of one colour have
    // disjoint 26-neighbourhoods, so they are decided independently (in
    // parallel) on the labels as updated by the earlier passes, then
    // applied. The result does not depend on the thread count.
    std::vector<Deferred> byColour[27];
    for (const auto& list : deferred) {
        for (const Deferred& c : list) {
            auto mod3 = [](int v) { return ((v % 3) + 3) % 3; };
            byColour[mod3(c.ijk.x()) + 3 * mod3(c.ijk.y()) + 9 * mod3(c.ijk.z())].push_back(c);
        }
    }
    const float barrier = float(1e-3 * mMls.voxelSize());
    tree::ValueAccessor<FloatTreeT> phiAcc(newPhi);
    tree::ValueAccessor<LabelTreeT> labAcc(newLabels);
    tree::ValueAccessor<BoolTreeT> maskAcc(freeze);
    for (const std::vector<Deferred>& pass : byColour) {
        if (pass.empty()) continue;
        std::vector<char> accept(pass.size(), 0);
        forEachIndex(pass.size(), [&](size_t i) {
            tree::ValueAccessor<const LabelTreeT> lab(newLabels);
            const Deferred& c = pass[i];
            accept[i] = this->topologyAllows(lab, c.ijk, c.from, c.to);
        });
        for (size_t i = 0; i < pass.size(); ++i) {
            const Deferred& c = pass[i];
            if (accept[i]) {
                phiAcc.setValueOnly(c.ijk, c.phi);
                labAcc.setValueOnly(c.ijk, LabelT(c.to));
            } else {
                // rejected: keep the label, barrier just inside its region
                phiAcc.setValueOnly(c.ijk, barrier);
                maskAcc.setValueOn(c.ijk);
            }
        }
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::bandMaintenance(const BoolTreeT& freeze)
{
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();
    const double dx = mMls.voxelSize();
    const float bg = phi.background();
    const float grow = float((mMls.halfWidth() - 1) * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // b. activate the inactive face neighbours of voxels with
    // phi < (halfWidth-1)*dx; label of the smallest-phi active face neighbour
    // (ties to the smaller label). Candidates are collected per leaf in
    // parallel on the unmodified trees, then applied serially in leaf order;
    // the result is the (phi, label) minimum per voxel, so it does not
    // depend on the order or the thread count.
    {
        struct Activation { Coord ijk; float phi; LabelT label; };
        tree::LeafManager<const FloatTreeT> leafs(phi);
        std::vector<std::vector<Activation>> found(leafs.leafCount());
        forEachIndex(leafs.leafCount(), [&](size_t i) {
            const auto& leaf = leafs.leaf(i);
            const auto* labLeaf = labels.probeConstLeaf(leaf.origin());
            tree::ValueAccessor<const FloatTreeT> phiAcc(phi);
            for (auto it = leaf.cbeginValueOn(); it; ++it) {
                if (!(*it < grow)) continue;
                const Coord p = it.getCoord();
                const LabelT l = labLeaf ? labLeaf->getValue(it.pos()) : labels.getValue(p);
                for (int f = 0; f < 6; ++f) {
                    const Coord q = p + offs[f];
                    if (!phiAcc.isValueOn(q)) found[i].push_back(Activation{q, *it, l});
                }
            }
        });
        // Serial merge. Every candidate voxel was inactive before the merge,
        // so an active one here was activated by an earlier candidate; its
        // phi temporarily holds the best candidate value.
        tree::ValueAccessor<FloatTreeT> phiW(phi);
        tree::ValueAccessor<LabelTreeT> labW(labels);
        std::vector<std::pair<Coord, LabelT>> added; // voxel, label before activation
        for (const auto& list : found) {
            for (const Activation& a : list) {
                float cur;
                if (phiW.probeValue(a.ijk, cur)) {
                    const LabelT curLabel = labW.getValue(a.ijk);
                    if (a.phi < cur || (a.phi == cur && a.label < curLabel)) {
                        phiW.setValueOnly(a.ijk, a.phi);
                        labW.setValueOnly(a.ijk, a.label);
                    }
                } else {
                    added.emplace_back(a.ijk, labW.getValue(a.ijk));
                    phiW.setValueOn(a.ijk, a.phi);
                    labW.setValueOn(a.ijk, a.label);
                }
            }
        }
        for (const auto& kv : added) {
            phiW.setValueOnly(kv.first, bg);
            // an activated voxel keeps its inactive label when the change
            // involves a protected label (topology preservation)
            const LabelT l = labW.getValue(kv.first);
            if (this->involvesProtected(static_cast<int32_t>(kv.second),
                    static_cast<int32_t>(l))) {
                labW.setValueOnly(kv.first, kv.second);
            }
        }
    }

    // c. redistance: projected and fixed voxels frozen, the rest solved with
    // signed reflection
    {
        MultiLevelSetRedistance<MultiLevelSetT> redist(mMls);
        redist.setTolerance(mTolerance);
        redist.setGrainSize(mGrainSize);
        redist.redistance(freeze);
    }

    // d. deactivate phi >= halfWidth*dx in both trees (labels kept)
    {
        tree::LeafManager<FloatTreeT> leafs(phi);
        forEachIndex(leafs.leafCount(), [&](size_t i) {
            auto& leaf = leafs.leaf(i);
            auto* labLeaf = labels.probeLeaf(leaf.origin());
            for (auto it = leaf.beginValueOn(); it; ++it) {
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
