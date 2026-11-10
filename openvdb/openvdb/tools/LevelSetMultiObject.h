// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObject.h
///
/// @brief Multi-object level sets: evolution of several disjoint objects
/// (and their contacts) by multi-object geodesic active contours (MOGAC)
/// on a single unsigned distance grid paired with an integer label grid,
/// following Lucas, Kazhdan and Taylor, MICCAI 2012 (see References below).
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
///
/// // labels are 16-bit by default; register that grid type before file I/O
/// tools::registerMultiLevelSetGridTypes();
/// io::File("pair.vdb").write({mls->phi(), mls->labels()});
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
/// - B. C. Lucas, M. Kazhdan, R. H. Taylor, "Multi-Object Geodesic Active
///   Contours (MOGAC): How to Represent and Manipulate Multiple Level Sets
///   with only a Label Mask and Distance Field", Medical Image Computing and
///   Computer Assisted Intervention (MICCAI), 2012. This tool implements the
///   representation and evolution described in this paper.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Grid.h>
#include <openvdb/codecs/ScalarCodec.h>// for registerMultiLevelSetGridTypes
#include <openvdb/io/Codec.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Mat4.h>
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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace lsmo_internal {

/// @name Named constants
/// Tolerances and tuning values of the multi-object level set tools.
/// "Fraction" and "voxels" values are multiples of the voxel size dx.
/// @{

/// Relative tolerance when checking that the phi background equals
/// halfWidth * dx (dimensionless).
constexpr double kBackgroundRelativeTolerance = 1e-6;
/// Epsilon below which a face-neighbor distance difference is treated as
/// zero when seeding the interface, as a fraction of dx.
constexpr double kSeedEpsilonFraction = 1e-6;
/// Default redistancing convergence tolerance, as a fraction of dx.
constexpr double kDefaultRedistanceToleranceFraction = 1e-4;
/// Extra iterations added to ceil(halfWidth * sqrt(3)) for the redistancing
/// iteration count (dimensionless count).
constexpr int kRedistanceExtraIterations = 2;
/// Safety multiplier on the redistancing iteration count that gives the
/// iteration cap (dimensionless).
constexpr int kRedistanceIterationCapFactor = 4;
/// Distance, in voxels, from a voxel to the interface it shares with a face
/// neighbor of a different label (the midpoint between the two voxels).
constexpr double kHalfVoxelDistance = 0.5;
/// Maximum interface motion per step, in voxels. Also the width of the
/// candidate band (phi <= this) in which a voxel may change label.
constexpr double kMaxInterfaceMotionVoxels = 0.5;
/// Default CFL number: fraction of the maximum interface motion a step may
/// use (dimensionless).
constexpr float kDefaultCfl = 0.999f;
/// Speed given to labels the speed functor reports as inactive: they retreat
/// at this fraction of the CFL-limited motion (dimensionless, near 1).
constexpr float kInactiveLabelSpeed = 0.999f;
/// Default cap on the magnitude of the curvature force (dimensionless).
constexpr float kDefaultMaxCurvatureForce = 10.0f;
/// Divisor in the explicit curvature time-step limit dx^2 / (6 w).
constexpr float kCurvatureDtDivisor = 6.0f;
/// Distance-field curvature correction 1 / (1 - psi*k) is applied only when
/// |psi*k| is below this (dimensionless).
constexpr float kCurvatureCorrectionMax = 0.5f;
/// Smallest magnitude of a field extremum that is still used to rescale the
/// field (field units).
constexpr float kRescaleMinMagnitude = 1e-4f;
/// Default minimum phi of protected-label voxels, as a fraction of dx.
constexpr double kDefaultMinimumPhiFraction = 1e-3;
/// Minimum phi of barrier and kinematic-stamp voxels, as a fraction of dx;
/// keeps a voxel on an obstacle surface from having phi = 0.
constexpr double kBarrierPhiFraction = 1e-3;
/// Inner bound of the obstacle shell that may join a protected label, as a
/// distance to the obstacle in voxels.
constexpr double kObstacleShellInnerVoxels = 0.5;
/// Outer bound (exclusive) of the same shell, in voxels.
constexpr double kObstacleShellOuterVoxels = 1.5;
/// Step, in voxels, of the search along the obstacle gradient for the
/// voxel a trapped protected label wraps onto.
constexpr double kWrapSearchStepVoxels = 0.5;
/// Reach of that search, in multiples of the half-width (voxels per voxel of
/// half-width).
constexpr double kWrapSearchReachBands = 4.0;
/// Most passes of the protected-label wrapping loop (count).
constexpr int kMaxWrapPasses = 8;
/// Safety factor applied to the kinematic step limit when shrinking dt
/// (dimensionless).
constexpr double kKinematicStepSafety = 0.99;
/// Largest factor by which one shrink iteration reduces dt (dimensionless).
constexpr double kKinematicStepShrinkMax = 0.5;
/// Most dt shrink iterations when limiting kinematic displacement (count).
constexpr int kMaxKinematicShrinkIterations = 32;
/// Relative tolerance on the end time of advect(t0, t1) (dimensionless).
constexpr double kAdvectTimeRelativeEpsilon = 1e-6;
/// Relative tolerance on the end time of one phase (parabolic half or
/// hyperbolic part) of a split step (dimensionless).
constexpr double kSplitPhaseRelativeEpsilon = 1e-6;
/// Primes of CoordHash, one per component (the usual spatial-hash primes).
constexpr uint64_t kCoordHashPrimeX = 73856093u;
constexpr uint64_t kCoordHashPrimeY = 19349669u;
constexpr uint64_t kCoordHashPrimeZ = 83492791u;
/// @}

/// Hash of a Coord for std::unordered_map and std::unordered_set: each
/// component times its own large prime, combined by xor.
struct CoordHash
{
    size_t operator()(const Coord& p) const noexcept
    {
        return size_t((uint64_t(uint32_t(p.x())) * kCoordHashPrimeX)
            ^ (uint64_t(uint32_t(p.y())) * kCoordHashPrimeY)
            ^ (uint64_t(uint32_t(p.z())) * kCoordHashPrimeZ));
    }
};

/// @brief True if the integer @a v is representable as a @c LabelT (any
/// mix of signed and unsigned types and widths).
template<typename LabelT, typename ValueT>
inline bool labelFits(ValueT v)
{
    static_assert(std::is_integral<LabelT>::value && std::is_integral<ValueT>::value,
        "labelFits: integer types only");
    using Limits = std::numeric_limits<LabelT>;
    if constexpr (std::is_signed<ValueT>::value) {
        if (v < 0) {
            return std::is_signed<LabelT>::value && int64_t(v) >= int64_t(Limits::min());
        }
    }
    return uint64_t(v) <= uint64_t(Limits::max());
}

} // namespace lsmo_internal


/// @brief Label tree of 16-bit labels, the default label tree of
/// MultiLevelSet (labels in [-32768, 32767]).
using Int16LabelTree = tree::Tree4<int16_t, 5, 4, 3>::Type;
/// @brief Grid of Int16LabelTree.
using Int16LabelGrid = Grid<Int16LabelTree>;

/// @brief Register the grid types of the multi-object tools that OpenVDB
/// does not register itself (Int16LabelGrid, with its I/O codec), so that
/// their grids can be read from and written to files.
/// @details Call it after openvdb::initialize() (and again after any
/// openvdb::uninitialize(), which clears the registries) and before file
/// I/O of a default MultiLevelSet's label grid. Idempotent and thread-safe
/// among its own callers: a type or codec already registered is left as it
/// is.
inline void registerMultiLevelSetGridTypes()
{
    using CodecT = codecs::ScalarCodec<Int16LabelGrid>;
    static std::mutex registryMutex;
    std::lock_guard<std::mutex> lock(registryMutex);
    if (!Int16LabelGrid::isRegistered()) Int16LabelGrid::registerGrid();
    if (!io::CodecRegistry::isRegistered(CodecT::name())) {
        io::CodecRegistry::registerCodec<CodecT>();
    }
}

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
/// time. The label tree may hold any integer type: Int16LabelTree (16-bit
/// labels) by default; use e.g. Int32Tree for label ids beyond 16 bits
/// (Int64Tree and UInt32Tree also work). Labels are int32_t in the
/// interface; where one is stored into the label tree (the factories,
/// LevelSetMultiObject::setKinematicLabel), a label the tree's value type
/// cannot represent throws ValueError rather than being truncated. File I/O
/// of an Int16LabelGrid needs registerMultiLevelSetGridTypes().
template<typename FloatTreeT = FloatTree, typename LabelTreeT = Int16LabelTree>
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
        if (!this->validateBackgrounds()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: " << kBackgroundMessage);
        }
        if (!this->validatePhi()) {
            OPENVDB_THROW(ValueError, "MultiLevelSet: " << kPhiMessage);
        }
        mPhi->setGridClass(GRID_UNKNOWN);
        mPhi->insertMeta("multi_level_set_role", StringMetadata("phi"));
        mLabels->insertMeta("multi_level_set_role", StringMetadata("labels"));
    }

    /// @brief Build a multi-object level set from a grid of labels.
    /// @details The label of a voxel is @a labels' value there (active or
    /// not); label 0 is background. Interface voxels (a voxel with a face
    /// neighbor of a different label) are searched among the active voxels
    /// of @a labels and their face neighbors and get phi = 0.5*dx (the voxel
    /// face midpoint). They are kept frozen while the rest of the band
    /// (interface voxels dilated by @a halfWidth over 26-neighbors) is
    /// redistanced; the interface values are not re-estimated with a subcell
    /// fix, because that gradient estimate needs a distance field, not this
    /// voxel staircase. Voxels left with phi >= halfWidth*dx are deactivated.
    /// Every inactive voxel kept in the trees holds its label from @a labels,
    /// so the label of a region's interior is known just outside the band.
    /// @a labels may hold any integer type (e.g. an Int32Grid for this
    /// object's 16-bit default label tree) with the node configuration of
    /// the label tree; every value (active or not, tiles included) must be
    /// representable as a @c LabelT.
    /// @param grainSize TBB grain size in leaves; zero or negative runs
    /// serially. The result does not depend on it.
    /// @throw ValueError if @a halfWidth < 2, the voxel size is not uniform,
    /// the label background is not 0, or a label of @a labels is outside the
    /// range of @c LabelT.
    template<typename InputGridT>
    static Ptr fromLabelGrid(const InputGridT& labels, int halfWidth = 3, int grainSize = 1);

    /// @brief Build a multi-object level set from signed distance grids, one
    /// per non-zero label.
    /// @details psi_l is the input grid of label l (inactive values read
    /// with their sign, so deep interiors are negative); the background
    /// label 0 has the implicit psi_0 = -min_l psi_l, i.e. it is the region
    /// outside every object. At each voxel the label is argmin psi (ties to
    /// the smaller label) and phi = (v_b - v_a)/2 for the two smallest values
    /// v_a <= v_b, so phi = |psi_l| near an isolated object. Candidates are
    /// the union of the input active topologies and their face neighbors;
    /// the band is the interface voxels dilated by @a halfWidth
    /// (26-neighbors), then reinitialized (interface voxels re-estimated
    /// with the Russo-Smereka subcell fix, the rest of the band solved from
    /// them; see MultiLevelSetRedistance::reinitialize) and trimmed to
    /// phi < halfWidth*dx. Inactive tiles and voxels kept in the label tree
    /// hold the label of the region they lie in, so the label of a region's
    /// interior is known just outside the band. Inactive tiles are labeled
    /// by the argmin at the tile origin, which is exact for disjoint inputs;
    /// where inputs overlap inside a tile, only voxels in leaves (of the
    /// band, its face shell or an input) are labeled per voxel.
    /// @param grainSize TBB grain size in leaves; zero or negative runs
    /// serially. The result does not depend on it.
    /// @throw ValueError if the map is empty, a grid is null or not a
    /// GRID_LEVEL_SET, a key is 0 or outside the range of @c LabelT, the
    /// transforms differ or are not a uniform scale, or @a halfWidth < 2.
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
    /// Seeds are the active phi voxels with a face neighbor on the other
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
    /// boundary in the band (e.g. it does not occur, or is outside the range
    /// of @c LabelT).
    /// @throw ValueError if @a halfWidth is not -1 and less than 2.
    template<typename InterruptT = util::NullInterrupter>
    typename FloatGridT::Ptr extractSignedDistance(int32_t label, int halfWidth = -1,
        int grainSize = 1, InterruptT* interrupt = nullptr) const;

    /// @brief Sorted distinct non-zero labels over all active voxels.
    std::vector<int32_t> labelSet() const
    {
        std::unordered_set<int32_t> s;
        for (auto it = mLabels->tree().cbeginValueOn(); it; ++it) {
            s.insert(static_cast<int32_t>(*it));
        }
        s.erase(0);
        std::vector<int32_t> v(s.begin(), s.end());
        std::sort(v.begin(), v.end());
        return v;
    }

    /// @brief Check the class invariant (identical active topology, phi
    /// finite and non-negative at active voxels, phi background
    /// halfWidth*dx and label background 0).
    /// @param message if non-null, receives a description of the first failure.
    bool validate(std::string* message = nullptr) const
    {
        if (!this->validateBackgrounds()) {
            if (message) *message = kBackgroundMessage;
            return false;
        }
        if (!mPhi->tree().hasSameTopology(mLabels->tree())) {
            if (message) *message = "phi and label trees have different active topology";
            return false;
        }
        if (!this->validatePhi()) {
            if (message) *message = kPhiMessage;
            return false;
        }
        return true;
    }

    /// @brief Dilate the active topology of both trees by @a iterations
    /// face-neighbor steps. New phi voxels get the background value; new
    /// label voxels copy the label of the active face neighbor with the
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
    /// face-neighbor steps, then call pruneInactive().
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

    static constexpr const char* kBackgroundMessage =
        "phi background must equal halfWidth * voxel size and the label background must be 0";
    static constexpr const char* kPhiMessage =
        "phi must be non-negative and finite at every active voxel";

    /// phi background == halfWidth*dx (relative tolerance
    /// lsmo_internal::kBackgroundRelativeTolerance), label background 0.
    bool validateBackgrounds() const
    {
        const double expected = double(mHalfWidth) * this->voxelSize();
        const double bg = double(mPhi->tree().background());
        return std::abs(bg - expected) <= lsmo_internal::kBackgroundRelativeTolerance * expected
            && mLabels->tree().background() == LabelT(0);
    }

    /// phi finite and non-negative at every active voxel.
    bool validatePhi() const
    {
        for (auto it = mPhi->tree().cbeginValueOn(); it; ++it) {
            const float v = *it;
            if (!std::isfinite(v) || v < 0.0f) return false;
        }
        return true;
    }

    typename FloatGridT::Ptr mPhi;
    typename LabelGridT::Ptr mLabels;
    int mHalfWidth;
};


namespace lsmo_internal {

/// Face-neighbor offsets, ordered (-x, +x, -y, +y, -z, +z).
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
/// @details A voxel p is a seed if differs(p, q) holds for some face neighbor
/// q (active or not). Its value is the Russo-Smereka subcell fix (Russo &
/// Smereka 2000; analyzed by du Chene, Min & Gibou 2008) applied to the
/// signed field psi of p's side: psi = -phi at p and at neighbors q with
/// !differs(p, q), +phi at neighbors with differs(p, q). Per axis
/// Delta_i = max(|psi(+1) - psi(-1)|/2, |psi(+1) - psi(0)|, |psi(0) - psi(-1)|,
/// 1e-6*dx) / dx and the seed is |psi_p| / sqrt(sum_i Delta_i^2); terms that
/// involve an inactive neighbor are left out (only band edges of finite
/// domains have them). A seed with no active face neighbor q with
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
    const double eps = kSeedEpsilonFraction * dx;
    const Coord* offs = faceOffsets();

    auto body = [&](const typename tree::LeafManager<FloatTreeT>::LeafRange& range) {
        DiffersOp op = differs; // per-task copy (may own value accessors)
        tree::ValueAccessor<const FloatTreeT> acc(snapshot);
        for (auto leafIt = range.begin(); leafIt; ++leafIt) {
            LeafT& leaf = *leafIt;
            auto* maskLeaf = seedMask.probeLeaf(leaf.origin());
            for (auto it = leaf.beginValueOn(); it; ++it) {
                const Coord p = it.getCoord();
                // psi at the six face neighbors, signed relative to p's side;
                // inactive neighbors are left out of the gradient.
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

/// One Jacobi update of the outer-layer solver at @a p (current value
/// @a old) from its active face neighbors; see solveOuterLayers().
template<typename FloatAccT, typename SeedAccT, typename DiffersOp>
inline float jacobiUpdate(const Coord& p, float old, FloatAccT& acc, SeedAccT& seedAcc,
    DiffersOp& op, double dx, bool signedReflection)
{
    const Coord* offs = faceOffsets();
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
    if (n == 0) return old;
    float upd = float(solveGodunov(abar, coef, n, dx));
    if (signedReflection) upd = std::max(upd, 0.0f);
    return std::min(old, upd);
}

/// @brief Outer-layer solver on the unsigned distance @a phi.
/// @details Active voxels of @a phi that are not active in @a seeds are reset
/// and solve |grad phi| = 1 by parallel Jacobi iteration with the seeds as
/// Dirichlet data: second-order one-sided upwind differences (Chopp 2001)
/// where two same-side upwind values are available (the outer one, q2, with
/// !differs(p, q2) and strictly smaller than the inner one) and first order
/// otherwise. Inactive neighbors are not used. Iteration stops when
/// max|delta phi| < @a tolerance, after @a maxIter iterations, or on
/// interrupt; voxels left unreached get the background of @a phi.
/// With @a signedReflection (used by the evolver, which freezes only the
/// voxels it just projected and must re-solve their other neighbors on both
/// sides of a moving interface) a face neighbor q of p
/// contributes +phi_q if !differs(p, q), -phi_q if differs(p, q) and q is a
/// seed, and nothing otherwise; the second-order term then also needs q on
/// p's side, and results are clamped to >= 0. This is exact for planar
/// interfaces whose frozen voxels lie on either side.
/// @param differs copied once per task; called as differs(p, q).
/// @param grainSize TBB grain size in leaves; zero or negative runs serially.
/// @param converged if non-null, set to whether max|delta phi| fell below
/// @a tolerance (rather than the iteration or the interrupt ending the solve).
/// @return the number of Jacobi iterations performed.
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp, typename InterruptT>
int solveOuterLayers(FloatTreeT& phi, const BoolTreeT& seeds, const DiffersOp& differs,
    double dx, float tolerance, int maxIter, int grainSize, InterruptT* interrupt,
    bool signedReflection = false, bool* converged = nullptr)
{
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
    using LeafManagerT = tree::LeafManager<FloatTreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;
    const float bg = phi.background();
    const bool serial = grainSize <= 0;

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
                    const float old = *it;
                    const float val = jacobiUpdate(it.getCoord(), old, acc, seedAcc, op, dx,
                        signedReflection);
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
    bool done = false;
    while (iter < maxIter) {
        if (util::wasInterrupted(interrupt)) break;
        ++iter;
        const float maxDelta = sweep();
        leafs.swapLeafBuffer(1, serial);
        if (maxDelta < tolerance) { done = true; break; }
    }
    if (converged) *converged = done;

    // Unreached voxels (no seed nearby, or interrupted) get the background.
    forEachLeaf(leafs, [&](typename LeafRangeT::Iterator& leafIt) {
        for (auto it = leafIt->beginValueOn(); it; ++it) {
            if (*it == kUnknownDistance) it.setValue(bg);
        }
    });
    return iter;
}

/// @brief solveOuterLayers() restricted to the voxels @a masks[i] of the
/// leaves @a leaves[i] of @a phi (active, and not in @a seeds).
/// @details Those voxels are reset and updated by the same Jacobi iteration
/// (and the same signed reflection) as in solveOuterLayers(); every other
/// active voxel keeps its value and acts as Dirichlet data, but only
/// voxels of @a seeds count as frozen across a label change. Iteration
/// stops when max|delta phi| over the solved voxels falls below
/// @a tolerance, after @a maxIter iterations, or on interrupt; unreached
/// voxels get the background. Only the given leaves are visited.
/// @param converged as in solveOuterLayers() (true if nothing is solved).
/// @return the number of Jacobi iterations performed (0 if none is solved).
template<typename FloatTreeT, typename BoolTreeT, typename MaskT, typename DiffersOp,
    typename InterruptT>
int solveOuterLayerLeaves(FloatTreeT& phi, const BoolTreeT& seeds,
    const std::vector<typename FloatTreeT::LeafNodeType*>& leaves,
    const std::vector<MaskT>& masks, const DiffersOp& differs, double dx, float tolerance,
    int maxIter, int grainSize, InterruptT* interrupt, bool signedReflection,
    bool* converged = nullptr)
{
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
    using LeafT = typename FloatTreeT::LeafNodeType;
    constexpr Index kSize = LeafT::SIZE;
    const float bg = phi.background();
    OPENVDB_ASSERT(leaves.size() == masks.size());
    if (converged) *converged = true;
    if (leaves.empty()) return 0;

    auto forEach = [&](const auto& op) {
        if (grainSize > 0) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size(), size_t(grainSize)),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) op(i);
                });
        } else {
            for (size_t i = 0; i < leaves.size(); ++i) op(i);
        }
    };

    // Reset the voxels to solve.
    forEach([&](size_t i) {
        for (auto it = masks[i].beginOn(); it; ++it) {
            leaves[i]->setValueOnly(it.pos(), kUnknownDistance);
        }
    });

    // Jacobi: new values of the solved voxels go to a scratch array per leaf,
    // then are written back.
    std::vector<float> next(leaves.size() * kSize);
    auto sweep = [&]() -> float {
        auto body = [&](const tbb::blocked_range<size_t>& r, float maxDelta) {
            DiffersOp op = differs; // per-task copy (may own value accessors)
            tree::ValueAccessor<const FloatTreeT> acc(phi);
            tree::ValueAccessor<const BoolTreeT> seedAcc(seeds);
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const LeafT& leaf = *leaves[i];
                float* out = next.data() + i * kSize;
                for (auto it = masks[i].beginOn(); it; ++it) {
                    const Index n = it.pos();
                    const float old = leaf.getValue(n);
                    const float val = jacobiUpdate(leaf.offsetToGlobalCoord(n), old, acc,
                        seedAcc, op, dx, signedReflection);
                    out[n] = val;
                    maxDelta = std::max(maxDelta, old - val);
                }
            }
            return maxDelta;
        };
        const tbb::blocked_range<size_t> all(0, leaves.size(),
            size_t(grainSize > 0 ? grainSize : 1));
        if (grainSize > 0) {
            return tbb::parallel_reduce(all, 0.0f, body,
                [](float a, float b) { return std::max(a, b); });
        }
        return body(all, 0.0f);
    };

    int iter = 0;
    bool done = false;
    while (iter < maxIter) {
        if (util::wasInterrupted(interrupt)) break;
        ++iter;
        const float maxDelta = sweep();
        forEach([&](size_t i) {
            const float* in = next.data() + i * kSize;
            for (auto it = masks[i].beginOn(); it; ++it) {
                leaves[i]->setValueOnly(it.pos(), in[it.pos()]);
            }
        });
        if (maxDelta < tolerance) { done = true; break; }
    }
    if (converged) *converged = done;

    // Unreached voxels (no seed or kept voxel nearby, or interrupted) get the
    // background.
    forEach([&](size_t i) {
        for (auto it = masks[i].beginOn(); it; ++it) {
            if (leaves[i]->getValue(it.pos()) == kUnknownDistance) {
                leaves[i]->setValueOnly(it.pos(), bg);
            }
        }
    });
    return iter;
}

/// @brief solveOuterLayerLeaves() on the voxels of the mask @a solve that are
/// active in @a phi and not in @a seeds (both masks may hold active tiles).
template<typename FloatTreeT, typename BoolTreeT, typename DiffersOp, typename InterruptT>
int solveOuterLayersMasked(FloatTreeT& phi, const BoolTreeT& seeds, const BoolTreeT& solve,
    const DiffersOp& differs, double dx, float tolerance, int maxIter, int grainSize,
    InterruptT* interrupt, bool signedReflection, bool* converged = nullptr)
{
    using LeafT = typename FloatTreeT::LeafNodeType;
    using MaskLeafT = typename BoolTreeT::LeafNodeType;
    using MaskT = typename MaskLeafT::NodeMaskType;
    std::vector<LeafT*> leaves;
    std::vector<MaskT> masks;
    // the voxels of a leaf in a mask: its leaf's, all (active tile) or none
    auto maskOf = [](tree::ValueAccessor<const BoolTreeT>& acc, const Coord& o) {
        if (const MaskLeafT* l = acc.probeConstLeaf(o)) return l->getValueMask();
        return MaskT(acc.isValueOn(o));
    };
    tree::ValueAccessor<const BoolTreeT> solveAcc(solve), seedAcc(seeds);
    for (auto it = phi.beginLeaf(); it; ++it) {
        MaskT m = maskOf(solveAcc, it->origin());
        if (m.isOff()) continue;
        m &= it->getValueMask();
        m -= maskOf(seedAcc, it->origin());
        if (m.isOff()) continue;
        leaves.push_back(&*it);
        masks.push_back(m);
    }
    return solveOuterLayerLeaves(phi, seeds, leaves, masks, differs, dx, tolerance, maxIter,
        grainSize, interrupt, signedReflection, converged);
}

} // namespace lsmo_internal


/// @brief Second-order label-aware redistancing of a MultiLevelSet.
///
/// @details Restores phi = distance to the voxel's own label boundary over
/// the active band while keeping the interface fixed:
/// -# Interface voxels (an active voxel with a face neighbor, active or
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
///    otherwise. Inactive neighbors are not used. Iteration stops when
///    max|delta phi| falls below the tolerance or after
///    ceil(halfWidth*sqrt(3)) + 2 iterations.
///
/// Only active phi voxels are written; topology, labels and inactive values
/// are untouched. Voxels with no path to a seed get the phi background.
/// Outer values are not clamped and may exceed the background
/// (halfWidth*dx); deactivating such voxels is left to band maintenance.
///
/// @pre The label tree holds the correct (deep-interior) label at the
/// inactive face neighbors of the band (inactive label voxels keep the
/// label of the region they lie in, see MultiLevelSet). A band-edge voxel
/// whose inactive neighbor reads the label background instead becomes a
/// spurious interface voxel.
///
/// @note FastSweeping.h was evaluated for reuse and rejected: its
/// update is first-order Godunov only, it sweeps in place (Gauss-Seidel), and
/// its boundary voxels are found from sign changes of the input field (or a
/// mask) through private initializers, so an all-positive field with
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
        : mMls(mls), mInterrupter(interrupt)
        , mTolerance(float(lsmo_internal::kDefaultRedistanceToleranceFraction
                           * mls.voxelSize()))
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
    /// with signed reflection: a face neighbor of the same label gives
    /// +phi, a frozen face neighbor of a different label gives -phi, other
    /// neighbors of a different label are ignored. Unlike redistance(),
    /// voxels next to a label change that are not in @a freeze are solved,
    /// not frozen. Unlike redistance(), the iteration runs until
    /// max|delta phi| falls below the tolerance (capped at iterationCap()
    /// iterations; see converged()), so the result is the converged solution
    /// of its Dirichlet data rather than a fixed number of sweeps.
    /// @return the number of Jacobi iterations performed.
    int redistance(const BoolTreeT& freeze)
    {
        FloatTreeT& phi = mMls.phi()->tree();
        const double dx = mMls.voxelSize();
        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");
        BoolTreeT seeds(freeze);
        seeds.voxelizeActiveTiles();
        seeds.topologyIntersection(phi);
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        const int iter = lsmo_internal::solveOuterLayers(phi, seeds, differs, dx,
            mTolerance, iterationCap(mMls.halfWidth()), mGrainSize, mInterrupter,
            /*signedReflection=*/true, &mConverged);
        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    /// @brief Incremental form of redistance(freeze): only the active voxels
    /// of @a dirty that are not active in @a freeze are reset and solved
    /// (same scheme, same signed reflection across frozen voxels, same
    /// convergence test); every other active voxel keeps its value and acts
    /// as Dirichlet data. The evolver passes every active voxel within
    /// halfWidth + 1 voxels of each voxel whose phi, label, activity or
    /// freeze state changed since phi was last redistanced. That region is
    /// a heuristic, not an equivalence: the Jacobi update only lowers
    /// values while its second-order term can raise one, so converged
    /// values also depend on the order in which voxels are reached, and the
    /// stopping test covers only the solved voxels (the full solve's covers
    /// the whole band). It reproduces redistance(freeze) to within the
    /// tolerance in the tested scenes for halfWidth <= 3; wider bands can
    /// differ by up to about 1e-2*dx. Use redistance(freeze) where the full
    /// solve's result is required.
    /// @return the number of Jacobi iterations performed (0 if nothing is
    /// solved).
    int redistance(const BoolTreeT& freeze, const BoolTreeT& dirty)
    {
        FloatTreeT& phi = mMls.phi()->tree();
        const double dx = mMls.voxelSize();
        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        const int iter = lsmo_internal::solveOuterLayersMasked(phi, freeze, dirty, differs,
            dx, mTolerance, iterationCap(mMls.halfWidth()), mGrainSize, mInterrupter,
            /*signedReflection=*/true, &mConverged);
        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    /// @brief redistance(freeze, dirty) with the voxels to solve given per
    /// leaf: @a masks[i] (active voxels of @a leaves[i], none in @a freeze).
    template<typename MaskT>
    int redistance(const BoolTreeT& freeze,
        const std::vector<typename FloatTreeT::LeafNodeType*>& leaves,
        const std::vector<MaskT>& masks)
    {
        FloatTreeT& phi = mMls.phi()->tree();
        if (mInterrupter) mInterrupter->start("Redistancing multi-object level set");
        const lsmo_internal::LabelDiffers<LabelTreeT> differs(mMls.labels()->tree());
        const int iter = lsmo_internal::solveOuterLayerLeaves(phi, freeze, leaves, masks,
            differs, mMls.voxelSize(), mTolerance, iterationCap(mMls.halfWidth()), mGrainSize,
            mInterrupter, /*signedReflection=*/true, &mConverged);
        if (mInterrupter) mInterrupter->end();
        return iter;
    }

    /// @brief Iteration cap of redistance(freeze) and its incremental forms:
    /// 4*(ceil(halfWidth*sqrt(3)) + 2), a safety limit. Narrow bands stop at
    /// the tolerance well before it; wide bands (halfWidth >= 4) with
    /// topology-preserving or kinematic labels can reach it, and the result
    /// is then less converged (converged() is false).
    static int iterationCap(int halfWidth)
    {
        return lsmo_internal::kRedistanceIterationCapFactor
            * (int(std::ceil(halfWidth * std::sqrt(3.0)))
               + lsmo_internal::kRedistanceExtraIterations);
    }
    /// @brief Whether the last redistance(freeze) (or incremental form)
    /// reached the tolerance before iterationCap() or an interrupt.
    bool converged() const { return mConverged; }

    /// @brief Interface seeding only, for per-label reconstruction.
    /// @details An active voxel p of @a out is a seed if some face neighbor
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
        const int maxIter = int(std::ceil(mMls.halfWidth() * std::sqrt(3.0)))
            + lsmo_internal::kRedistanceExtraIterations;

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
    bool mConverged = true;
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
/// Every outside seed with a face neighbor (active or not) of a third label
/// (neither @a label nor L(q)) is rebuilt by a local Godunov extension of
/// psi: per axis the smallest known psi among the two face neighbors
/// (-phi at region seeds, +value at other outside seeds and at rebuilt seeds
/// once solved), then sum_i (psi - a_i)^2 = dx^2, clamped to
/// [phi_q, min(B, background)] with B the upper bound over the active region
/// voxels p within the 5x5x5 neighborhood of q. Jacobi sweeps over just
/// these seeds until no value changes by more than 1e-4*dx (at most 8). A
/// seed with no known neighbor keeps phi_q. psi is not smooth at junction
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

    const float tol = float(kDefaultRedistanceToleranceFraction * dx);
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
/// input value; phi is 0.5*dx at voxels with a face neighbor of a different
/// label and @a bg elsewhere. Copied once per task.
template<typename LabelTreeT>
struct LabelGridSampler
{
    LabelGridSampler(const LabelTreeT& t, double dx, float bg)
        : acc(t), half(float(kHalfVoxelDistance * dx)), background(bg) {}
    LabelGridSampler(const LabelGridSampler& o)
        : acc(o.acc.tree()), half(o.half), background(o.background) {}

    /// The input label, in the input's own type (range-checked by the caller).
    typename LabelTreeT::ValueType label(const Coord& p) { return acc.getValue(p); }
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
/// -# Interface voxels: voxels of @a candidates with a face neighbor of a
///    different sampled label.
/// -# Band: interface voxels dilated by @a halfWidth (26-neighbors); every
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
                        const auto l = s.label(p);
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
    if (!lsmo_internal::labelFits<LabelT>(label)) {
        // no voxel can hold the label: all outside, without truncating it
        typename FloatGridT::Ptr grid = FloatGridT::create(bg);
        grid->setTransform(mPhi->transform().copy());
        grid->setGridClass(GRID_LEVEL_SET);
        return grid;
    }
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

    // 2. output tree over the band, initialized with phi
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
    const int maxIter = int(std::ceil(halfWidth * std::sqrt(3.0)))
        + lsmo_internal::kRedistanceExtraIterations;
    const float tol = float(lsmo_internal::kDefaultRedistanceToleranceFraction * dx);
    lsmo_internal::solveOuterLayers(*out, seeds, sideOp, dx, tol, maxIter,
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
template<typename InputGridT>
typename MultiLevelSet<FloatTreeT, LabelTreeT>::Ptr
MultiLevelSet<FloatTreeT, LabelTreeT>::fromLabelGrid(const InputGridT& labels, int halfWidth,
    int grainSize)
{
    using InputTreeT = typename InputGridT::TreeType;
    using InputT = typename InputTreeT::ValueType;
    static_assert(std::is_integral<InputT>::value,
        "MultiLevelSet::fromLabelGrid: the input grid must have integer values");
    if (halfWidth < 2) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: halfWidth must be at least 2");
    }
    if (!labels.transform().hasUniformScale()) {
        OPENVDB_THROW(ValueError,
            "MultiLevelSet::fromLabelGrid: transform must have uniform voxel size");
    }
    if (labels.background() != InputT(0)) {
        OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: label background must be 0");
    }
    for (auto it = labels.tree().cbeginValueAll(); it; ++it) {
        if (!lsmo_internal::labelFits<LabelT>(*it)) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromLabelGrid: label " << +*it << " at "
                << it.getCoord() << " is outside the range of the label tree's value type");
        }
    }
    const double dx = labels.transform().voxelSize()[0];

    // Keep the input's structure and values (deep-interior labels), inactive.
    typename LabelTreeT::Ptr lab(new LabelTreeT(labels.tree()));
    for (auto it = lab->beginValueOn(); it; ++it) it.setValueOff();

    BoolTreeT candidates(labels.tree(), false, TopologyCopy());
    candidates.voxelizeActiveTiles();
    tools::dilateActiveValues(candidates, 1, NN_FACE, IGNORE_TILES);

    return build(labels.transform(), lab, candidates,
        lsmo_internal::LabelGridSampler<InputTreeT>(labels.tree(), dx, float(halfWidth * dx)),
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
        if (!lsmo_internal::labelFits<LabelT>(kv.first)) {
            OPENVDB_THROW(ValueError, "MultiLevelSet::fromSignedGrids: label " << kv.first
                << " is outside the range of the label tree's value type");
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
/// math::CurvatureStencil (0 center, 1-6 faces, 7-18 edges), or -1.
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
/// center voxel through sampleAtCenter(), which caches the result per center
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
            mScalarCache.invalidate();
            mVectorCache.invalidate();
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

    /// @brief Value of @a grid at the world position of the center voxel.
    /// If the transform of @a grid equals that of the stencil's grids, the
    /// index spaces coincide and the voxel value is read directly (exactly
    /// what BoxSampler returns at an integer index); otherwise, or if
    /// @a forceSampling is true, the value is BoxSampler-interpolated. The
    /// value is cached per grid until the stencil moves to another voxel,
    /// and each grid is read through a value accessor kept by the stencil
    /// (several grids, e.g. per-label fields, are cached side by side).
    /// @a grid must outlive the stencil.
    float sampleAtCenter(const FloatGrid& grid, bool forceSampling = false) const
    {
        return this->sample(grid, mScalarCache, forceSampling);
    }
    /// @brief Vector-valued version of sampleAtCenter().
    Vec3f sampleAtCenter(const Vec3SGrid& grid, bool forceSampling = false) const
    {
        return this->sample(grid, mVectorCache, forceSampling);
    }

    /// @brief psi at offset (i,j,k) from the center; the 19 points have at
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
    /// Cache entry of a sampled grid: the grid, a value accessor to its
    /// tree, whether the grid shares the stencil's transform, and the value
    /// at the current center voxel.
    template<typename GridT>
    struct FieldEntry
    {
        const GridT* grid = nullptr;
        bool aligned = false;
        std::unique_ptr<tree::ValueAccessor<const typename GridT::TreeType>> acc;
        bool valid = false;
        typename GridT::ValueType value{};
    };

    /// Cache of up to kMaxFields grids keyed by grid address, so that fields
    /// of several labels sampled at one voxel do not evict each other.
    template<typename GridT>
    struct FieldCache
    {
        static constexpr int kMaxFields = 8;
        std::array<FieldEntry<GridT>, kMaxFields> entries;
        int size = 0, next = 0;

        void invalidate() { for (int i = 0; i < size; ++i) entries[i].valid = false; }

        FieldEntry<GridT>& find(const GridT& grid, const math::Transform& xform)
        {
            for (int i = 0; i < size; ++i) {
                if (entries[i].grid == &grid) return entries[i];
            }
            // new grid: append, or replace the oldest entry when full
            const int i = size < kMaxFields ? size++ : (next++ % kMaxFields);
            FieldEntry<GridT>& e = entries[i];
            e.grid = &grid;
            e.acc.reset(new tree::ValueAccessor<const typename GridT::TreeType>(grid.tree()));
            e.aligned = (grid.transform() == xform);
            e.valid = false;
            return e;
        }
    };

    template<typename GridT>
    typename GridT::ValueType sample(const GridT& grid, FieldCache<GridT>& caches,
        bool forceSampling) const
    {
        FieldEntry<GridT>& cache = caches.find(grid, mTransform);
        if (!cache.valid) {
            if (cache.aligned && !forceSampling) {
                cache.value = cache.acc->getValue(mCenter);
            } else {
                const Vec3d w = mTransform.indexToWorld(mCenter);
                const Vec3R idx = grid.transform().worldToIndex(w);
                cache.value = tools::BoxSampler::sample(*cache.acc, idx);
            }
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
/// curvature k_x of the level set through the voxel center (falling back to
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
    void setMaxCurvatureForce(float m = lsmo_internal::kDefaultMaxCurvatureForce)
    {
        mMaxCurvatureForce = m;
    }

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
            const float normMin = (std::abs(minValue) > lsmo_internal::kRescaleMinMagnitude)
                ? 1.0f / std::abs(minValue) : 1.0f;
            const float normMax = (std::abs(maxValue) > lsmo_internal::kRescaleMinMagnitude)
                ? 1.0f / std::abs(maxValue) : 1.0f;
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

    /// @brief Test hook: when true, fields are always BoxSampler-interpolated,
    /// even if their transform equals that of the level set (default false:
    /// such fields are read directly, with identical results).
    void setForceSampling(bool force) { mForceSampling = force; }

    /// @brief Advection velocity (world units), sampled with BoxSampler unless
    /// its transform equals that of the level set (then read directly).
    /// Shared by all labels without their own field (see below).
    /// @throw ValueError if @a velocity is null.
    void setAdvection(Vec3SGrid::ConstPtr velocity, float weight)
    {
        if (!velocity) OPENVDB_THROW(ValueError, "MultiObjectForces: null advection grid");
        mAdvection = velocity;
        mAdvectionWeight = weight;
    }

    /// @brief Advection velocity and weight of label @a label only,
    /// overriding the shared field for that label; sampled exactly like the
    /// shared field.
    /// @throw ValueError if @a velocity is null.
    void setAdvection(int32_t label, Vec3SGrid::ConstPtr velocity, float weight)
    {
        if (!velocity) OPENVDB_THROW(ValueError, "MultiObjectForces: null advection grid");
        mLabelAdvection[label] = LabelAdvection{std::move(velocity), weight};
    }

    /// @brief Remove the advection override of @a label (it uses the shared
    /// field again, if any).
    void clearLabelAdvection(int32_t label) { mLabelAdvection.erase(label); }

    void setInactiveLabels(const std::set<int32_t>& labels)
    {
        mInactive = std::unordered_set<int32_t>(labels.begin(), labels.end());
    }
    void setFixedLabels(const std::set<int32_t>& labels)
    {
        mFixed = std::unordered_set<int32_t>(labels.begin(), labels.end());
    }

    /// The internal rescaled pressure copy (null if none was set).
    FloatGrid::ConstPtr pressure() const { return mPressure; }

    /// @brief dpsi_l/dt at @a ijk, the sum of hyperbolic() and parabolic();
    /// the stencil must have been moved to (@a ijk, @a label).
    float operator()(const Coord& ijk, int32_t label,
        const LabelAwareStencil<MultiLevelSetT>& s) const
    {
        OPENVDB_ASSERT(ijk == s.center());
        (void)ijk; // fields are sampled at s.center()
        if (!isActive(label)) return lsmo_internal::kInactiveLabelSpeed;
        if (!mFixed.empty() && mFixed.count(label)) return 0.0f;
        float dneg[3], dpos[3];
        s.upwindDifferences(dneg, dpos);
        const float kappa = this->curvatureTerm(s);
        return -this->advectionTerm(label, s, dneg, dpos) + kappa
            + this->pressureTerm(s, dneg, dpos);
    }

    /// @brief The first-order-in-space part of operator(): advection,
    /// pressure and the speed of inactive labels.
    float hyperbolic(const Coord& ijk, int32_t label,
        const LabelAwareStencil<MultiLevelSetT>& s) const
    {
        OPENVDB_ASSERT(ijk == s.center());
        (void)ijk;
        if (!isActive(label)) return lsmo_internal::kInactiveLabelSpeed;
        if (!mFixed.empty() && mFixed.count(label)) return 0.0f;
        float dneg[3], dpos[3];
        s.upwindDifferences(dneg, dpos);
        return -this->advectionTerm(label, s, dneg, dpos) + this->pressureTerm(s, dneg, dpos);
    }

    /// @brief The curvature part of operator() (with its interface
    /// correction and clamp); 0 for inactive and fixed labels.
    float parabolic(const Coord& ijk, int32_t label,
        const LabelAwareStencil<MultiLevelSetT>& s) const
    {
        OPENVDB_ASSERT(ijk == s.center());
        (void)ijk;
        if (!isActive(label)) return 0.0f;
        if (!mFixed.empty() && mFixed.count(label)) return 0.0f;
        return this->curvatureTerm(s);
    }

    /// True for labels declared fixed. Label 0 is NOT fixed here: the
    /// evolver treats it as passive and never asks the functor for its speed.
    bool isFixed(int32_t label) const
    {
        return label != 0 && !mFixed.empty() && mFixed.count(label) != 0;
    }
    bool isActive(int32_t label) const
    {
        return mInactive.empty() || mInactive.count(label) == 0;
    }

    /// dx^2/(6 w) for curvature weight w > 0, else +infinity.
    float stableDt(float dx) const
    {
        return mCurvatureWeight > 0.0f
            ? dx * dx / (lsmo_internal::kCurvatureDtDivisor * mCurvatureWeight)
                                       : std::numeric_limits<float>::infinity();
    }
    /// Stable time step of parabolic() alone: stableDt() (the curvature is
    /// the only term with a stability limit).
    float parabolicStableDt(float dx) const { return this->stableDt(dx); }

private:
    /// Clamped interface curvature speed (0 without curvature).
    float curvatureTerm(const LabelAwareStencil<MultiLevelSetT>& s) const
    {
        if (mCurvatureWeight == 0.0f) return 0.0f;
        // Interface curvature: the voxel's level set has mean curvature
        // k_x; the interface, at signed distance -psi along the normal,
        // has k = k_x / (1 - psi*k_x) for a sphere (a first-order
        // correction otherwise). Falls back to k_x when |psi*k_x| >= 0.5.
        float kn = s.meanCurvatureNormGrad(); // k_x * |grad psi|
        const float g = s.gradient().length();
        if (g > 0.0f) {
            const float pk = s.template getValue<0, 0, 0>() * kn / g; // psi*k_x
            if (std::abs(pk) < lsmo_internal::kCurvatureCorrectionMax) kn /= (1.0f - pk);
        }
        const float kappa = mCurvatureWeight * kn;
        return std::max(-mMaxCurvatureForce, std::min(mMaxCurvatureForce, kappa));
    }

    /// Upwinded advection term v.grad psi (sampled at the stencil center).
    float advectionTerm(int32_t label, const LabelAwareStencil<MultiLevelSetT>& s,
        const float dneg[3], const float dpos[3]) const
    {
        float advection = 0.0f;
        const Vec3SGrid* advField = mAdvection.get();
        float advWeight = mAdvectionWeight;
        if (!mLabelAdvection.empty()) {
            const auto it = mLabelAdvection.find(label);
            if (it != mLabelAdvection.end()) {
                advField = it->second.field.get();
                advWeight = it->second.weight;
            }
        }
        if (advField) {
            const Vec3f vel = s.sampleAtCenter(*advField, mForceSampling);
            for (int a = 0; a < 3; ++a) {
                const float f = advWeight * vel[a];
                if (f > 0.0f) advection += f * dneg[a];
                else if (f < 0.0f) advection += f * dpos[a];
            }
        }
        return advection;
    }

    /// Godunov-upwinded pressure term (sampled at the stencil center).
    float pressureTerm(const LabelAwareStencil<MultiLevelSetT>& s,
        const float dneg[3], const float dpos[3]) const
    {
        if (!mPressure) return 0.0f;
        const float force = mPressureWeight * s.sampleAtCenter(*mPressure, mForceSampling);
        float gsPos = 0.0f, gsNeg = 0.0f;
        for (int a = 0; a < 3; ++a) {
            const float nMax = std::max(dneg[a], 0.0f), nMin = std::min(dneg[a], 0.0f);
            const float pMax = std::max(dpos[a], 0.0f), pMin = std::min(dpos[a], 0.0f);
            gsPos += nMax * nMax + pMin * pMin;
            gsNeg += pMax * pMax + nMin * nMin;
        }
        if (force > 0.0f) return -force * std::sqrt(gsPos);
        if (force < 0.0f) return -force * std::sqrt(gsNeg);
        return 0.0f;
    }

    float mCurvatureWeight = 0.0f;
    float mMaxCurvatureForce = lsmo_internal::kDefaultMaxCurvatureForce;
    FloatGrid::ConstPtr mPressure;
    float mPressureWeight = 0.0f;
    Vec3SGrid::ConstPtr mAdvection;
    float mAdvectionWeight = 0.0f;
    struct LabelAdvection
    {
        Vec3SGrid::ConstPtr field;
        float weight = 0.0f;
    };
    // hashed: looked up per voxel, never iterated
    std::unordered_map<int32_t, LabelAdvection> mLabelAdvection;
    bool mForceSampling = false;
    std::unordered_set<int32_t> mInactive, mFixed;
};


////////////////////////////////////////


namespace lsmo_internal {

/// @brief Whether a speed functor is split: it provides
/// hyperbolic(ijk, label, stencil) and parabolic(ijk, label, stencil)
/// besides operator().
template<typename SpeedT, typename = void>
struct IsSplitSpeed: std::false_type {};
template<typename SpeedT>
struct IsSplitSpeed<SpeedT,
    std::void_t<decltype(&SpeedT::hyperbolic), decltype(&SpeedT::parabolic)>>
    : std::true_type {};

/// Whether a speed functor provides parabolicStableDt(dx).
template<typename SpeedT, typename = void>
struct HasParabolicStableDt: std::false_type {};
template<typename SpeedT>
struct HasParabolicStableDt<SpeedT,
    std::void_t<decltype(std::declval<const SpeedT&>().parabolicStableDt(0.0f))>>
    : std::true_type {};

} // namespace lsmo_internal


/// @brief Digital adjacency pair (foreground, background) used by
/// topology-preserving evolution and the multi-label mesher.
enum class MultiObjectConnectivity { FG26_BG6, FG6_BG26, FG18_BG6, FG6_BG18 };


namespace lsmo_internal {

/// Adjacency tables of the 3x3x3 neighborhood, bit i <-> (dx,dy,dz) with
/// i = (dx+1) + 3(dy+1) + 9(dz+1): adj[a][i] holds the positions other than
/// the center that are a-adjacent to i (a = 0: 6, 1: 18, 2: 26), and
/// star[a] the a-neighbors of the center (N*_6, N*_18, N*_26).
struct SimplePointTables
{
    uint32_t adj[3][27];
    uint32_t star[3];
    SimplePointTables()
    {
        auto coord = [](int i) { return Coord(i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1); };
        const uint32_t center = uint32_t(1) << 13;
        for (int a = 0; a < 3; ++a) {
            star[a] = 0;
            for (int i = 0; i < 27; ++i) {
                adj[a][i] = 0;
                for (int j = 0; j < 27; ++j) {
                    const Coord d = coord(i) - coord(j);
                    const int d2 = d.x() * d.x() + d.y() * d.y() + d.z() * d.z();
                    if (d2 >= 1 && d2 <= a + 1) adj[a][i] |= uint32_t(1) << j;
                }
                adj[a][i] &= ~center;
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

/// Bertrand-Malandain topological number T_n(x, X) of the center x for the
/// positions @a X of the 26-neighborhood (center excluded): the number of
/// n-components of the geodesic neighborhood G_n(x, X) = N^k_n(x, X), with
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

/// @brief Whether the center of a 3x3x3 neighborhood is a simple point of
/// the foreground region @a inMask (27 bits, bit i <-> (dx,dy,dz) with
/// i = (dx+1) + 3(dy+1) + 9(dz+1); the center bit is ignored) under the
/// adjacency pair @a conn: adding or removing it changes neither the
/// region's nor its complement's topology (no new or lost components,
/// tunnels or cavities). Decided by T_fg = 1 and T_bg = 1 (Bertrand and
/// Malandain 1994).
inline bool isSimplePoint(uint32_t inMask, MultiObjectConnectivity conn)
{
    const uint32_t all = (uint32_t(1) << 27) - 1, center = uint32_t(1) << 13;
    const uint32_t fg = inMask & all & ~center, bg = ~inMask & all & ~center;
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
///    a fixed label and has phi <= 0.5*dx or a face neighbor (active or
///    not) of a different label is projected: these are the only voxels
///    whose label can change in one step (see Time step), so the update is
///    restricted to them. Its candidates are its own label and the labels
///    of its face neighbors, minus fixed labels. Label 0 is passive: it is a
///    candidate only if it is the voxel's or a face neighbor's label or the
///    voxel touches a fixed label, and its value is the complement
///    v_0 = -min v_l over the non-zero candidates, so object/background
///    interfaces move at the object's full speed (and a voxel next to a
///    retreating fixed label falls back to label 0). The speed functor is
///    called (on time-n data through a LabelAwareStencil) only for non-zero
///    candidates; labels with isActive() false get dpsi/dt = +0.999, so
///    they slowly retreat from their neighbors.
/// -# <b>Time step.</b> dt = min(maxDt, 0.5*cfl*dx/max|F|, speed.stableDt(dx))
///    (see Operator splitting below for split speed functors),
///    so an interface moves at most half a voxel per step. The
///    half-voxel cap is what makes the projected set sufficient: since
///    psi_l >= phi at every voxel, no voxel outside it can change label by
///    projection (orphan cleanup can still relabel), so no separate
///    propagation pass is needed.
/// -# <b>Projection.</b> v_l = psi_l + dt*F_l with psi_l = -phi for the own
///    label and +phi otherwise; the new label is the smallest v (ties to the
///    smaller label) and phi = |v_1 - v_2|/2 over the two smallest values, or
///    |v_1| with one candidate. Results are written to an auxiliary leaf
///    buffer of each tree (time-n data stays readable in the leaf buffers
///    until both trees swap buffers together).
/// -# <b>Topology preservation</b> (opt-in): with
///    setTopologyPreservingLabels(), a projected change a -> b that involves
///    a protected label is deferred and applied in 27 colored passes
///    (coordinates mod 3) only if the voxel is a simple point
///    (lsmo_internal::isSimplePoint, under setConnectivity()) of every
///    protected label in {a, b} on the labels as updated by earlier passes;
///    a rejected voxel keeps label a with phi = 1e-3*dx and stays frozen.
///    Orphan cleanup and band activation never change a protected label's
///    region except by simple changes, so its components, cavities and
///    tunnels are preserved. With no protected label nothing changes.
/// -# <b>Orphan cleanup</b>: a voxel whose label occurs nowhere in its
///    26-neighborhood takes the label and phi of its smallest-phi active
///    neighbor (fixed labels excluded) and is redistanced (not frozen).
/// -# <b>Band maintenance</b>: after the buffer swap, inactive face
///    neighbors of active voxels with phi < (halfWidth-1)*dx are activated
///    (phi = background, label of the smallest-phi active face neighbor);
///    MultiLevelSetRedistance::redistance(freeze) keeps the projected voxels
///    and the fixed-label voxels and solves all others with signed
///    reflection, so voxels the interface just passed are re-solved from
///    both sides; voxels with phi >= halfWidth*dx are deactivated in
///    both trees (labels kept, so the interior label stays known);
///    pruneInactive(). Builds with OPENVDB_ENABLE_ASSERTS check validate()
///    after every step.
///    The redistance iterates to the tolerance (see
///    MultiLevelSetRedistance::iterationCap), so its result is the converged
///    solution of the frozen values. By default it is incremental: every
///    write of the step (projection, deferred changes, kinematic stamps,
///    orphan fixes, band activation and deactivation, the phi floor) and
///    every change of the frozen set records its voxels, and only the
///    leaves within halfWidth + 1 voxels of a recorded voxel are re-solved;
///    the others keep their values as Dirichlet data. This assumes that
///    only the evolver changes the pair between steps; the first step, and the first after
///    a change of the topology-preserving or kinematic labels, the phi
///    floor or the tolerance, re-solve the whole band, as does every step
///    with setFullRedistance(true).
///    The halfWidth + 1 region is a heuristic, not a proof of equivalence:
///    the Jacobi update only lowers values while its second-order upwind
///    term can raise a value when a neighbor drops, so a converged result
///    also depends on the order in which voxels are reached (which the kept
///    values change), and the incremental stopping test covers only the
///    re-solved leaves while the full one covers the whole band. In the
///    tested scenes it reproduces the full solve to within the tolerance
///    for halfWidth <= 3; wider bands can differ by up to about 1e-2*dx.
///    setFullRedistance(true) is exact.
///
/// Voxels of fixed labels never change label or phi and no label can
/// capture them; a step with no non-zero candidate and no kinematic label
/// anywhere changes nothing.
///
/// <b>Operator splitting.</b> With a split speed functor (one that also
/// provides hyperbolic() and parabolic(), see SpeedT concept) and
/// setSplitting(true) (the default), a step is split only when the
/// curvature limit binds: when parabolicStableDt < min(maxDt,
/// 0.5*cfl*dx/max|F_h|), with F_h the hyperbolic speeds on the state at the
/// start of the step (evaluated only if parabolicStableDt < maxDt).
/// Otherwise it is the unsplit update above, unchanged. A split step is a
/// Strang step: parabolic(dt/2), hyperbolic(dt), parabolic(dt/2), with
/// dt = min(maxDt, 0.5*cfl*dx/max|F_h|) (and the kinematic limit), so the
/// hyperbolic terms (advection, pressure, inactive-label retreat) are not
/// held to the much smaller curvature limit.
/// Each parabolic half takes n = ceil((dt/2)/parabolicStableDt) equal
/// substeps. Every substep and the hyperbolic part are full updates as
/// above (steps 1-6: projection with passive label 0 and fixed walls,
/// topology preservation, orphan cleanup, the phi floor, incremental
/// redistance, band maintenance), each evaluated on the state the previous
/// one left and each limited to half a voxel of motion: if its own speeds
/// require it, an update takes a shorter time step and the part continues
/// with more updates until its time is covered. An update whose speeds are
/// all zero is skipped. (If the first parabolic half raises max|F_h|, the
/// hyperbolic part takes more than one update, so that step is slightly
/// asymmetric.) Kinematic labels are stamped once per step, at t_{n+1}, by
/// its last update (the earlier ones treat them as walls at their t_n
/// position). A step with no parabolic part (parabolicStableDt +inf, e.g.
/// curvature weight 0) is never split. The interrupter is checked between
/// parabolic substeps and before and after the hyperbolic part; on
/// interrupt the step ends after the current update with a valid state,
/// kinematic labels stamped at the time reached, and returns the time
/// advanced: the parabolic time done if the hyperbolic part has not run,
/// else dt (the parabolic part then lags by less than dt/2).
///
/// <b>Kinematic labels</b> (setKinematicLabel()) are rigid obstacles whose
/// region is prescribed: a signed distance grid of the object in its rest
/// pose and a rigid motion of time. They are treated like fixed labels by
/// the steps above (never captured, frozen, no speed evaluation). After the
/// deferred changes and before orphan cleanup, each step re-stamps them at
/// its end time t_{n+1}: every voxel near the moved object samples the rest
/// pose distance psi at motion(t)^-1 x; where psi < 0 it gets the kinematic
/// label with phi = |psi| (at least 1e-3*dx; active and frozen if
/// |psi| < halfWidth*dx); a
/// voxel of the label with psi >= 0 (the object left it) becomes passive
/// label 0 and is redistanced; other voxels with psi < halfWidth*dx are
/// activated and get phi = min(phi, psi) (frozen if psi <= dx). Claims on
/// label 0 and on unprotected labels always win; a claim on a voxel of a
/// topology-preserving label goes through the same simple-point test (27
/// colored passes, setConnectivity() rule) as other protected changes,
/// and a rejected claim leaves the voxel with the protected label (phi
/// 1e-3*dx, frozen): the obstacle lacks that voxel this step and is
/// re-stamped from its motion in the next. Such trapped voxels are then
/// wrapped outward: each moves to the first voxel o along the object's
/// outward normal with psi(o) >= 0.5*dx that is neither kinematic nor fixed
/// (o joins the protected label if that is a simple change, as do outer
/// shell voxels 0.5*dx <= psi < 1.5*dx within two voxels of the trapped one
/// that touch the protected label; then the object claims the trapped
/// voxel if that is simple), repeated up to 8 passes per
/// step; both changes keep every protected label's topology. The wrap is
/// not always complete: where an object exits a topology-protected label
/// (for example through a wall the label cannot open), voxels of that label
/// can stay inside the object, so the object is dented there (its region
/// can lose volume, gain cavities or, under 6-adjacency, split) until the
/// protected label lets them go. The time step
/// is also limited so that the moved
/// object's rest bounding box corners move at most 0.5*cfl*dx per step.
///
/// @par SpeedT concept
/// @code
/// float operator()(const Coord& ijk, int32_t label,
///                  const LabelAwareStencil<MultiLevelSetT>& s) const; // dpsi/dt
/// bool  isFixed(int32_t label) const;
/// bool  isActive(int32_t label) const;
/// float stableDt(float dx) const; // +inf if none
/// // optional, all three for operator splitting (operator() = their sum):
/// float hyperbolic(const Coord& ijk, int32_t label,
///                  const LabelAwareStencil<MultiLevelSetT>& s) const;
/// float parabolic(const Coord& ijk, int32_t label,
///                 const LabelAwareStencil<MultiLevelSetT>& s) const;
/// float parabolicStableDt(float dx) const; // +inf if none (else stableDt)
/// @endcode
/// operator(), hyperbolic() and parabolic() must be thread-safe; parabolic()
/// is not called for inactive labels (their retreat is hyperbolic). All
/// quantities are in world units.
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
        , mTolerance(float(lsmo_internal::kDefaultRedistanceToleranceFraction * mls.voxelSize()))
    {
    }

    /// @brief CFL number (default 0.999).
    /// @throw ValueError if @a cfl is not positive.
    void setCFL(float cfl)
    {
        if (!(cfl > 0.0f)) OPENVDB_THROW(ValueError, "LevelSetMultiObject: CFL must be positive");
        mCfl = cfl;
    }
    /// @brief TBB grain size in leaves; zero or negative runs serially.
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }
    /// @brief Redistance tolerance in world units (default 1e-4*dx).
    void setRedistanceTolerance(float tolWorld) { mTolerance = tolWorld; mPrimed = false; }
    /// @brief Re-solve the whole band at every step (default false: only the
    /// leaves near the voxels the step changed are re-solved; see the class
    /// documentation). In the tested scenes the results agree to within the
    /// redistance tolerance for halfWidth <= 3 and differ by up to about
    /// 1e-2*dx beyond; true makes the result exact.
    void setFullRedistance(bool on) { mFullRedistance = on; }
    bool fullRedistance() const { return mFullRedistance; }
    /// @brief Strang splitting of a split speed functor's hyperbolic and
    /// parabolic parts (default true; see the class documentation): a step
    /// is split only when speed.parabolicStableDt(dx) is below the
    /// hyperbolic dt min(maxDt, 0.5*cfl*dx/max|F_h|); otherwise it is the
    /// unsplit update. Ignored for a functor that only provides operator().
    void setSplitting(bool on) { mSplitting = on; }
    /// Whether steps are split: setSplitting() is on and the speed functor
    /// is split (a step is then split if its parabolicStableDt is finite).
    bool splitting() const { return mSplitting && kSplitSpeed; }
    /// @brief Labels whose digital topology is preserved (default
    /// none): a label change a -> b at a voxel is applied only if the voxel is
    /// a simple point of every protected label in {a, b}.
    /// @throw ValueError if a label is kinematic, or if label 0 is listed
    /// while kinematic labels exist (passive label 0 takes over the voxels
    /// an obstacle leaves, which cannot be protected).
    void setTopologyPreservingLabels(std::set<int32_t> labels)
    {
        for (const int32_t l : labels) {
            if (this->isKinematic(l)) {
                OPENVDB_THROW(ValueError, "LevelSetMultiObject: kinematic label " << l
                    << " cannot be topology-preserving");
            }
        }
        if (labels.count(0) && !mKinematic.empty()) {
            OPENVDB_THROW(ValueError,
                "LevelSetMultiObject: label 0 cannot be topology-preserving with kinematic labels");
        }
        mProtected = std::unordered_set<int32_t>(labels.begin(), labels.end());
        mPrimed = false;
    }
    /// @brief Minimum phi, as a fraction of the voxel size, kept at the
    /// active voxels of topology-preserving labels and at their face
    /// neighbors across a protected interface, after every change the
    /// evolver makes (only while some label is topology-preserving; other
    /// evolutions, the factories and extraction are unaffected). It keeps
    /// interface vertices of protected labels apart, so their meshes stay
    /// manifold. Default 1e-3.
    /// @throw ValueError if @a fractionOfDx is negative or not finite.
    void setMinimumPhi(double fractionOfDx)
    {
        if (!(fractionOfDx >= 0.0) || !std::isfinite(fractionOfDx)) {
            OPENVDB_THROW(ValueError, "LevelSetMultiObject: invalid minimum phi");
        }
        mMinPhiFraction = fractionOfDx;
        mPrimed = false;
    }
    double minimumPhi() const { return mMinPhiFraction; }
    /// @brief The phi floor in world units: minimumPhi()*dx if some label is
    /// topology-preserving, else 0 (pass it as
    /// MultiObjectMeshSettings::minimumPhi).
    double phiFloor() const
    {
        return mProtected.empty() ? 0.0 : mMinPhiFraction * mMls.voxelSize();
    }
    /// @brief Whether the phi floor holds (see setMinimumPhi()); true if no
    /// label is topology-preserving.
    bool validateTopologyFloor() const;
    /// @brief Adjacency pair of topology preservation (default FG26_BG6).
    void setConnectivity(MultiObjectConnectivity conn) { mConnectivity = conn; }

    /// @brief Make @a label kinematic: a rigid obstacle whose region at time
    /// t is {x : psi(motion(t)^-1 x) < 0}, with psi the level set @a sdf of
    /// the object in its rest pose (world space, BoxSampler-interpolated,
    /// any transform) and motion(t) a rigid transform in the row-vector
    /// convention of math::Mat4d::transform (x_t = motion(t).transform(x)).
    /// See the class documentation. Voxels already labeled @a label that
    /// lie in or near the object at the first step are re-stamped by it.
    /// Kinematic voxels never take voxels of fixed or other kinematic
    /// labels, nor change their phi.
    /// @throw ValueError if @a label is 0 or outside the range of the label
    /// tree's value type, @a sdf is null, @a motion is empty, @a label is
    /// topology-preserving or label 0 is.
    void setKinematicLabel(int32_t label, FloatGrid::ConstPtr sdf,
        std::function<math::Mat4d(double)> motion)
    {
        if (label == 0) OPENVDB_THROW(ValueError, "LevelSetMultiObject: kinematic label 0");
        if (!lsmo_internal::labelFits<LabelT>(label)) {
            OPENVDB_THROW(ValueError, "LevelSetMultiObject: kinematic label " << label
                << " is outside the range of the label tree's value type");
        }
        if (!sdf) OPENVDB_THROW(ValueError, "LevelSetMultiObject: null kinematic sdf");
        if (this->isProtected(label)) {
            OPENVDB_THROW(ValueError, "LevelSetMultiObject: topology-preserving label " << label
                << " cannot be kinematic");
        }
        if (this->isProtected(0)) {
            OPENVDB_THROW(ValueError,
                "LevelSetMultiObject: no kinematic labels while label 0 is topology-preserving");
        }
        if (!motion) OPENVDB_THROW(ValueError, "LevelSetMultiObject: empty kinematic motion");
        Kinematic k;
        k.sdf = std::move(sdf);
        k.motion = std::move(motion);
        const CoordBBox b = k.sdf->evalActiveVoxelBoundingBox();
        if (!b.empty()) {
            const Vec3d lo = k.sdf->indexToWorld(b.min()), hi = k.sdf->indexToWorld(b.max());
            for (int c = 0; c < 8; ++c) {
                k.corners[c] = Vec3d(c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1],
                    c & 4 ? hi[2] : lo[2]);
            }
            k.hasBox = true;
        }
        mKinematic[label] = std::move(k);
        mPrimed = false;
    }
    /// @brief Make @a label an ordinary label again (its voxels stay).
    void clearKinematicLabel(int32_t label) { mKinematic.erase(label); mPrimed = false; }
    /// @brief Time of the current state (set by advect(); step() advances it
    /// by the time step taken). Kinematic labels are stamped at this time
    /// plus the step.
    void setTime(double t) { mTime = t; }
    double time() const { return mTime; }

    float cfl() const { return mCfl; }
    int grainSize() const { return mGrainSize; }
    float redistanceTolerance() const { return mTolerance; }
    /// The topology-preserving labels, in increasing order (a copy: the
    /// evolver keeps them hashed for its per-voxel lookups).
    std::set<int32_t> topologyPreservingLabels() const
    {
        return std::set<int32_t>(mProtected.begin(), mProtected.end());
    }
    MultiObjectConnectivity connectivity() const { return mConnectivity; }

    /// @brief Peak bytes of the evolver's own working memory during the last
    /// step: the auxiliary leaf buffers of both trees plus the per-leaf
    /// scratch (candidate lists, masks) that is kept and reused across steps.
    /// Temporaries of the redistance pass are not included. 0 before the
    /// first step.
    size_t scratchMemUsage() const { return mPeakScratchBytes; }
    /// @brief Jacobi iterations of the last step's redistance (0 before the
    /// first step or if nothing was re-solved).
    int redistanceIterations() const { return mRedistanceIterations; }
    /// @brief Redistances since construction that did not reach the
    /// tolerance (MultiLevelSetRedistance::converged() false). The evolver
    /// does not pass its interrupter to the redistance, so these are the
    /// solves stopped by the iteration cap
    /// (MultiLevelSetRedistance::iterationCap); normally 0, but wide bands
    /// with topology-preserving or kinematic labels can reach it.
    size_t redistanceCapHits() const { return mRedistanceCapHits; }

    /// @brief One forward Euler step of at most @a maxDt (world time), or
    /// one Strang-split step (see the class documentation).
    /// @return the time step taken; min(@a maxDt, speed.stableDt) without
    /// changes if nothing can move, 0 if @a maxDt <= 0; less than the step
    /// if a split step was interrupted (see the class documentation).
    float step(float maxDt);

    /// @brief Step from @a t0 to @a t1, checking the interrupter before each
    /// step (and within split steps). On interrupt the state after the last
    /// completed step or sub-update is kept (it is valid) and time() is the
    /// time reached.
    /// @return the number of steps taken.
    size_t advect(double t0, double t1)
    {
        if (mInterrupter) mInterrupter->start("Evolving multi-object level set");
        size_t n = 0;
        double t = t0;
        const double eps = lsmo_internal::kAdvectTimeRelativeEpsilon
            * std::max(1.0, std::abs(t1));
        while (t1 - t > eps) {
            if (util::wasInterrupted(mInterrupter)) break;
            mTime = t;
            const float dt = this->step(float(t1 - t));
            if (!(dt > 0.0f)) break;
            t += dt;
            ++n;
            if (mInterrupted) break; // a split step stopped early
        }
        if (mInterrupter) mInterrupter->end();
        return n;
    }

private:
    /// Own label + 6 face labels.
    static constexpr int kMaxCandidates = 7;
    static constexpr bool kSplitSpeed = lsmo_internal::IsSplitSpeed<SpeedT>::value;

    /// The part of the speed a sub-update evaluates: operator(), or
    /// hyperbolic() or parabolic() of a split functor.
    enum class Term { Full, Hyperbolic, Parabolic };
    /// Largest |speed| over the candidates of the last evaluate() and
    /// whether any voxel has a non-zero candidate.
    struct Evaluation
    {
        float maxSpeed = 0.0f;
        bool any = false;
    };

    /// A candidate label and its speed.
    using Candidate = std::pair<int32_t, float>;

    /// Candidate labels of one voxel being built.
    struct CandidateSet
    {
        std::array<int32_t, kMaxCandidates> label;
        int n = 0;
        bool zero = false;
    };

    /// One projected voxel with a non-zero candidate: its n candidates
    /// (labels with speeds) follow those of the previous entry in
    /// LeafData::cand; zero says whether passive label 0 participates. Its
    /// own label and phi are read from the time-n leaf buffers.
    struct VoxelEntry
    {
        uint16_t offset = 0;
        uint8_t n = 0;
        bool zero = false;
    };
    static_assert(FloatTreeT::LeafNodeType::SIZE <= 65536, "leaf offsets must fit 16 bits");

    struct LeafData
    {
        std::vector<VoxelEntry> entries;
        std::vector<Candidate> cand;
        float maxSpeed = 0.0f;
        bool any = false;
    };

    /// A projected label change involving a protected label.
    struct Deferred { Coord ijk; int32_t from, to; float phi; };
    /// An orphan relabeled from its smallest-phi neighbor.
    struct Fix { Coord ijk; LabelT label; float phi; };
    /// A band voxel to activate.
    struct Activation { Coord ijk; float phi; LabelT label; };

    using BoolLeafT = typename BoolTreeT::LeafNodeType;

    /// Voxel bit masks per leaf origin: the change masks of incremental
    /// redistancing. Much smaller than a mask tree, whose internal nodes
    /// cost about as much as those of the pair. Lookups (probe) may run
    /// concurrently; insertions are serial.
    struct LeafMasks
    {
        using MaskT = typename BoolLeafT::NodeMaskType;
        static constexpr Int32 kDimMask = Int32(BoolLeafT::DIM - 1);

        static Coord originOf(const Coord& p)
        {
            return Coord(p.x() & ~kDimMask, p.y() & ~kDimMask, p.z() & ~kDimMask);
        }
        MaskT* probe(const Coord& origin)
        {
            auto it = map.find(origin);
            return it == map.end() ? nullptr : &it->second;
        }
        const MaskT* probe(const Coord& origin) const
        {
            auto it = map.find(origin);
            return it == map.end() ? nullptr : &it->second;
        }
        /// Set the bit of voxel @a p (serial).
        void setOn(const Coord& p) { map[originOf(p)].setOn(BoolLeafT::coordToOffset(p)); }
        /// An entry at every leaf of @a phi (serial; existing bits kept).
        void touch(const FloatTreeT& phi)
        {
            for (auto leaf = phi.cbeginLeaf(); leaf; ++leaf) map[leaf->origin()];
        }
        /// All bits off, one entry per leaf of @a phi (serial).
        void reset(const FloatTreeT& phi)
        {
            for (auto it = map.begin(); it != map.end();) {
                if (phi.probeConstLeaf(it->first)) { it->second.setOff(); ++it; }
                else it = map.erase(it);
            }
            this->touch(phi);
        }
        size_t memUsage() const
        {
            // nodes (value plus about two pointers and a hash) and buckets
            return map.size() * (sizeof(typename decltype(map)::value_type)
                + 3 * sizeof(void*)) + map.bucket_count() * sizeof(void*);
        }

        std::unordered_map<Coord, MaskT, lsmo_internal::CoordHash> map;
    };

    /// Phi, label and activity of one voxel, for change tracking.
    struct VoxelState { float phi = 0.0f; LabelT label = LabelT(0); bool on = false; };

    /// Change tracking for incremental redistancing (serial writers): every
    /// write to the pair goes through write(), which compares the voxel
    /// before and after and marks it dirty if it changed.
    struct ChangeLog
    {
        explicit ChangeLog(LeafMasks& dirtyMasks): dirty(dirtyMasks) {}

        template<typename PhiAccT, typename LabAccT>
        static VoxelState state(PhiAccT& phi, LabAccT& lab, const Coord& p)
        {
            VoxelState s;
            s.on = phi.probeValue(p, s.phi);
            s.label = lab.getValue(p);
            return s;
        }
        /// Run @a w (which writes voxel @a p through @a phi and @a lab) and
        /// mark @a p if its phi (while active), label or activity changed.
        template<typename PhiAccT, typename LabAccT, typename WriteT>
        void write(PhiAccT& phi, LabAccT& lab, const Coord& p, const WriteT& w)
        {
            const VoxelState a = state(phi, lab, p);
            w();
            const VoxelState b = state(phi, lab, p);
            if (a.on != b.on || a.label != b.label || (a.on && a.phi != b.phi)) {
                dirty.setOn(p);
            }
        }

        LeafMasks& dirty;
    };

    template<typename T>
    static size_t vectorBytes(const std::vector<T>& v) { return v.capacity() * sizeof(T); }
    template<typename T>
    static size_t vectorBytes(const std::vector<std::vector<T>>& v)
    {
        size_t n = v.capacity() * sizeof(std::vector<T>);
        for (const auto& w : v) n += w.capacity() * sizeof(T);
        return n;
    }

    /// Per-leaf scratch owned by the evolver and reused across steps: lists
    /// keep their storage (see recycle()), the per-leaf arrays grow only
    /// when the leaf count grows, and mask leaves are recycled.
    struct Scratch
    {
        std::vector<LeafData> data;
        std::vector<std::vector<Deferred>> deferred;
        /// projected and fixed-label voxels (redistance freeze mask)
        BoolTreeT freeze{false};
        /// incremental redistancing: voxels whose phi, label or activity
        /// changed since the last redistance
        LeafMasks dirty;
        /// frozen active voxels of the last redistance
        LeafMasks prevSeeds;
        std::vector<std::unique_ptr<BoolLeafT>> freeLeaves;

        /// Empty @a v for reuse. Slots reused by other leaves would keep
        /// growing to the largest list they ever held, so a list holding
        /// more than twice what it used last time (plus some slack) is
        /// reallocated at its last size.
        template<typename T>
        static void recycle(std::vector<T>& v)
        {
            constexpr size_t kSlack = 16;
            if (v.capacity() > 2 * v.size() + kSlack) {
                std::vector<T> w;
                w.reserve(v.size());
                v.swap(w);
            } else {
                v.clear();
            }
        }

        /// At least @a n empty lists.
        template<typename T>
        static void resetLists(std::vector<std::vector<T>>& v, size_t n)
        {
            if (v.size() < n) v.resize(n);
            for (auto& w : v) recycle(w);
        }

        void resetData(size_t n)
        {
            if (data.size() < n) data.resize(n);
            for (LeafData& d : data) {
                recycle(d.entries);
                recycle(d.cand);
                d.maxSpeed = 0.0f;
                d.any = false;
            }
        }

        /// Make @a freeze an inactive (false) mask with one leaf per leaf
        /// of @a phi, recycling the leaves of the previous step.
        void resetFreeze(const FloatTreeT& phi) { resetMask(freeze, phi); }

        /// Make @a mask an inactive (false) mask with one leaf per leaf of
        /// @a phi, recycling its leaves.
        void resetMask(BoolTreeT& mask, const FloatTreeT& phi)
        {
            std::vector<BoolLeafT*> stolen;
            mask.stealNodes(stolen);
            for (BoolLeafT* leaf : stolen) freeLeaves.emplace_back(leaf);
            mask.clear();
            for (auto leaf = phi.cbeginLeaf(); leaf; ++leaf) {
                mask.addLeaf(this->takeLeaf(leaf->origin()));
            }
        }

        /// An inactive mask leaf at @a origin (recycled if possible).
        BoolLeafT* takeLeaf(const Coord& origin)
        {
            if (freeLeaves.empty()) return new BoolLeafT(origin, false, false);
            BoolLeafT* leaf = freeLeaves.back().release();
            freeLeaves.pop_back();
            leaf->setOrigin(origin);
            leaf->fill(false, false);
            return leaf;
        }

        size_t memUsage() const
        {
            size_t n = vectorBytes(data);
            for (const LeafData& d : data) n += vectorBytes(d.entries) + vectorBytes(d.cand);
            n += vectorBytes(deferred);
            n += size_t(freeze.memUsage()) + dirty.memUsage() + prevSeeds.memUsage();
            n += freeLeaves.capacity() * sizeof(std::unique_ptr<BoolLeafT>)
                + freeLeaves.size() * sizeof(BoolLeafT);
            return n;
        }
    };

    /// Double buffering of both trees through tree::LeafManager auxiliary
    /// buffers: time n+1 is written to aux buffer kNext of each leaf while
    /// the leaf buffers keep time n, then both trees swap buffers together.
    /// Further aux buffers per leaf (e.g. a Runge-Kutta stage) fit after
    /// kNext.
    struct PairBuffers
    {
        static constexpr size_t kNext = 1;
        static constexpr size_t kAuxPerLeaf = 1;
        using PhiBufferT = typename FloatTreeT::LeafNodeType::Buffer;
        using LabelBufferT = typename LabelTreeT::LeafNodeType::Buffer;

        PairBuffers(FloatTreeT& phiTree, LabelTreeT& labelTree, bool serial)
            : phi(phiTree, kAuxPerLeaf, serial), labels(labelTree, kAuxPerLeaf, serial)
            , mSerial(serial)
        {
            // label leaf of each phi leaf (the same index unless the label
            // tree has leaves where phi has none)
            bool same = labels.leafCount() == phi.leafCount();
            for (size_t i = 0; same && i < phi.leafCount(); ++i) {
                same = labels.leaf(i).origin() == phi.leaf(i).origin();
            }
            if (same) return;
            std::unordered_map<Coord, size_t, lsmo_internal::CoordHash> index;
            index.reserve(labels.leafCount());
            for (size_t j = 0; j < labels.leafCount(); ++j) index[labels.leaf(j).origin()] = j;
            mLabelIndex.resize(phi.leafCount());
            for (size_t i = 0; i < phi.leafCount(); ++i) {
                const auto it = index.find(phi.leaf(i).origin());
                OPENVDB_ASSERT(it != index.end());
                mLabelIndex[i] = it->second;
            }
        }

        PhiBufferT& phiBuffer(size_t i, size_t b) { return phi.getBuffer(i, b); }
        LabelBufferT& labelBuffer(size_t i, size_t b)
        {
            return labels.getBuffer(mLabelIndex.empty() ? i : mLabelIndex[i], b);
        }
        /// Swap aux buffer @a b with the leaf buffers of both trees.
        void swap(size_t b)
        {
            phi.swapLeafBuffer(b, mSerial);
            labels.swapLeafBuffer(b, mSerial);
        }
        void release()
        {
            phi.removeAuxBuffers();
            labels.removeAuxBuffers();
        }
        size_t memUsage() const
        {
            const size_t phiBytes = sizeof(PhiBufferT)
                + FloatTreeT::LeafNodeType::SIZE * sizeof(float);
            const size_t labelBytes = sizeof(LabelBufferT)
                + LabelTreeT::LeafNodeType::SIZE * sizeof(LabelT);
            return phi.auxBufferCount() * phiBytes + labels.auxBufferCount() * labelBytes
                + mLabelIndex.capacity() * sizeof(size_t);
        }

        tree::LeafManager<FloatTreeT> phi;
        tree::LeafManager<LabelTreeT> labels;
    private:
        std::vector<size_t> mLabelIndex;
        bool mSerial;
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

    /// Fixed labels of the speed functor and kinematic labels (walls).
    bool isFixed(int32_t l) const
    {
        return l != 0 && (mSpeed.isFixed(l) || this->isKinematic(l));
    }
    bool isKinematic(int32_t l) const
    {
        return !mKinematic.empty() && mKinematic.count(l) != 0;
    }

    /// A kinematic label: rest-pose level set, motion, rest bounding box
    /// corners (world) and the index box stamped last.
    struct Kinematic
    {
        FloatGrid::ConstPtr sdf;
        std::function<math::Mat4d(double)> motion;
        std::array<Vec3d, 8> corners;
        bool hasBox = false;
        CoordBBox lastBox;
    };

    /// Largest displacement of a kinematic object's rest box between times
    /// @a t0 and @a t1 (world units).
    double kinematicDisplacement(double t0, double t1) const
    {
        double d = 0.0;
        for (const auto& kv : mKinematic) {
            if (!kv.second.hasBox) continue;
            const math::Mat4d a = kv.second.motion(t0), b = kv.second.motion(t1);
            for (const Vec3d& c : kv.second.corners) {
                d = std::max(d, (b.transform(c) - a.transform(c)).length());
            }
        }
        return d;
    }

    void stampKinematic(double t, BoolTreeT& freeze);

    /// Whether @a p (own label @a own) is subject to the phi floor: a voxel
    /// of a protected label or a face neighbor of one.
    template<typename AccT>
    bool floorApplies(AccT& lab, const Coord& p, int32_t own) const
    {
        if (this->isProtected(own)) return true;
        const Coord* offs = lsmo_internal::faceOffsets();
        for (int f = 0; f < 6; ++f) {
            const int32_t l = static_cast<int32_t>(lab.getValue(p + offs[f]));
            if (l != own && this->isProtected(l)) return true;
        }
        return false;
    }
    /// Raise phi to phiFloor() wherever floorApplies (no-op without
    /// protected labels); raised voxels are marked dirty (the masks need an
    /// entry at every phi leaf).
    void enforcePhiFloor();

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

    void applyDeferred(const std::vector<std::vector<Deferred>>& deferred,
        FloatTreeT& phi, LabelTreeT& labels, BoolTreeT& freeze);

    /// The 27 colored simple-point passes of applyDeferred(): each change
    /// is decided on the labels as updated by the earlier passes, then
    /// @a accept(change) or @a reject(change) is called (serially).
    template<typename AcceptT, typename RejectT>
    void coloredPasses(const std::vector<std::vector<Deferred>>& deferred,
        const LabelTreeT& labels, const AcceptT& accept, const RejectT& reject) const;

    /// Add @a l to the candidates @a c: label 0 and fixed labels only make
    /// passive label 0 participate (a fixed label can never be captured).
    void addCandidate(CandidateSet& c, int32_t l) const
    {
        if (l == 0 || this->isFixed(l)) { c.zero = true; return; } // passive 0 / wall
        for (int i = 0; i < c.n; ++i) if (c.label[i] == l) return;
        c.label[c.n++] = l; // at most 7 distinct labels
    }

    float evalSpeed(StencilT& s, const Coord& ijk, int32_t label, Term term) const
    {
        if (!mSpeed.isActive(label)) {
            // inactive labels retreat (a hyperbolic term)
            return term == Term::Parabolic ? 0.0f : lsmo_internal::kInactiveLabelSpeed;
        }
        s.moveTo(ijk, label);
        if constexpr (kSplitSpeed) {
            if (term == Term::Hyperbolic) return mSpeed.hyperbolic(ijk, label, s);
            if (term == Term::Parabolic) return mSpeed.parabolic(ijk, label, s);
        }
        return mSpeed(ijk, label, s);
    }

    /// Stable time step of the parabolic part (speed.stableDt if the
    /// functor has no parabolicStableDt).
    float parabolicStableDt() const
    {
        const float dx = float(mMls.voxelSize());
        if constexpr (lsmo_internal::HasParabolicStableDt<SpeedT>::value) {
            return mSpeed.parabolicStableDt(dx);
        } else {
            return mSpeed.stableDt(dx);
        }
    }
    /// Steps 1-2 of the class documentation on the current state: freeze
    /// mask, projected set, candidates and their @a term speeds (scratch).
    Evaluation evaluate(Term term);
    /// @a dt limited so that interfaces move at most half a voxel.
    float motionLimit(const Evaluation& e, float dt) const;
    /// @a dt limited so that kinematic objects move at most half a voxel.
    float kinematicLimit(float dt) const;
    /// Steps 4-6 with time step @a dt, on the last evaluate(); kinematic
    /// labels are stamped at @a stampTime if @a stamp.
    void update(float dt, bool any, bool stamp, double stampTime);
    /// Strang step: parabolic(dt/2), hyperbolic(dt), parabolic(dt/2), with
    /// @a h the hyperbolic evaluation of the current state.
    float splitStep(float maxDt, const Evaluation& h);
    /// Sub-updates of @a term over @a duration, each of at most @a target
    /// (and the motion limit); the last stamps the kinematic labels at
    /// @a stampTime if @a stampAtEnd. If @a interruptible, the interrupter
    /// is checked after every sub-update but the last (and after the last
    /// too if @a checkAfterLast); on interrupt @a interrupted is set and the
    /// phase stops. @return the time covered.
    double splitPhase(Term term, double duration, float target, bool stampAtEnd,
        double stampTime, bool interruptible, bool checkAfterLast, bool& interrupted);
    /// End a split step interrupted after @a elapsed: stamp the kinematic
    /// labels at mTime + @a elapsed, advance the time. @return @a elapsed.
    float finishInterrupted(float elapsed);

    /// Two-smallest projection of @a e (own label @a own, time-n
    /// @a phiN) with candidates @a cand; false if @a e has no non-zero
    /// candidate.
    static bool project(const VoxelEntry& e, int32_t own, float phiN, const Candidate* cand,
        float dt, int32_t& label, float& phi)
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
            const int32_t l = cand[i].first;
            const float psi = (l == own) ? -phiN : phiN;
            const float v = psi + dt * cand[i].second;
            consider(l, v);
            minV = std::min(minV, v);
        }
        if (e.zero) consider(0, -minV); // passive label 0: complement
        label = l1;
        phi = (count > 1) ? std::abs(0.5f * (v1 - v2)) : std::abs(v1);
        return true;
    }

    void cleanOrphans(FloatTreeT& phi, LabelTreeT& labels, BoolTreeT& freeze);
    void bandMaintenance(const BoolTreeT& freeze);
    /// Redistance only near the changes recorded since the last redistance.
    /// @return the number of Jacobi iterations.
    int incrementalRedistance(MultiLevelSetRedistance<MultiLevelSetT>& redist,
        const BoolTreeT& freeze);
#ifdef OPENVDB_ENABLE_ASSERTS
    /// Debug check: every voxel whose phi (while active), label or activity
    /// differs from the state after the last redistance is marked dirty.
    void checkChangesRecorded() const;
    std::unique_ptr<FloatTreeT> mCheckPhi;
    std::unique_ptr<LabelTreeT> mCheckLabels;
#endif
    /// Keep the frozen active voxels of this redistance (Scratch::prevSeeds).
    void rememberSeeds(const BoolTreeT& freeze);

    MultiLevelSetT& mMls;
    const SpeedT& mSpeed;
    InterruptT* mInterrupter;
    // hashed for the per-voxel lookups; stampKinematic visits the kinematic
    // labels in increasing order (overlapping objects: the lower label wins)
    std::unordered_set<int32_t> mProtected;
    std::unordered_map<int32_t, Kinematic> mKinematic;
    double mTime = 0.0;
    MultiObjectConnectivity mConnectivity = MultiObjectConnectivity::FG26_BG6;
    double mMinPhiFraction = lsmo_internal::kDefaultMinimumPhiFraction;
    float mCfl = lsmo_internal::kDefaultCfl;
    int mGrainSize = 1;
    float mTolerance;
    bool mFullRedistance = false;
    bool mSplitting = true;
    /// whether the last step() stopped early on an interrupt
    bool mInterrupted = false;
    int mRedistanceIterations = 0;
    size_t mRedistanceCapHits = 0;
    /// whether the pair was last redistanced by this evolver (incremental
    /// redistancing needs a redistanced state outside the dirty region)
    bool mPrimed = false;
    Scratch mScratch;
    size_t mPeakScratchBytes = 0;
};


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::step(float maxDt)
{
    mInterrupted = false;
    if (!(maxDt > 0.0f)) return 0.0f;
    mPeakScratchBytes = 0;
    if (this->splitting()) {
        // split only if the parabolic limit is below the hyperbolic dt
        const float pStable = this->parabolicStableDt();
        if (pStable > 0.0f && pStable < maxDt) { // false for +inf
            const Evaluation h = this->evaluate(Term::Hyperbolic);
            if (h.any && pStable < this->motionLimit(h, maxDt)) {
                return this->splitStep(maxDt, h);
            }
        }
    }

    // 1-2. projected set, candidates and speeds, all on time-n data
    const Evaluation e = this->evaluate(Term::Full);

    // 3. time step: interfaces move at most half a voxel, so only projected
    // voxels can change label
    float dt = std::min(maxDt, mSpeed.stableDt(float(mMls.voxelSize())));
    if (!e.any && mKinematic.empty()) {
        mTime += dt;
        return dt; // no non-zero candidate anywhere: nothing moves
    }
    dt = this->kinematicLimit(this->motionLimit(e, dt));

    // 4-6. projection, deferred changes, kinematic labels at t_{n+1}, orphan
    // cleanup, band maintenance
    this->update(dt, e.any, /*stamp=*/true, mTime + double(dt));
    mTime += dt;
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
typename LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::Evaluation
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::evaluate(Term term)
{
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();
    const double dx = mMls.voxelSize();
    const float halfDx = float(lsmo_internal::kMaxInterfaceMotionVoxels * dx);
    const Coord* offs = lsmo_internal::faceOffsets();

    // The label tree needs a leaf wherever phi has one (normally a no-op).
    {
        tree::ValueAccessor<LabelTreeT> acc(labels);
        for (auto leaf = phi.cbeginLeaf(); leaf; ++leaf) acc.touchLeaf(leaf->origin());
    }

    // Freeze mask: projected voxels and fixed-label voxels.
    Scratch& scratch = mScratch;
    scratch.resetFreeze(phi);
    BoolTreeT& freeze = scratch.freeze;
    // Change masks of incremental redistancing (they keep the changes made
    // after the last redistance)
    scratch.dirty.touch(phi);

    // 1-2. projected set, candidates and speeds, all on time-n data
    tree::LeafManager<const FloatTreeT> leafs(phi);
    const size_t leafCount = leafs.leafCount();
    scratch.resetData(leafCount);
    std::vector<LeafData>& data = scratch.data;
    constexpr Index kLeafSize = FloatTreeT::LeafNodeType::SIZE;
    forEachIndex(leafCount, [&](size_t i) {
        const auto& leaf = leafs.leaf(i);
        LeafData& d = data[i];
        auto* maskLeaf = freeze.probeLeaf(leaf.origin());
        tree::ValueAccessor<const LabelTreeT> lab(labels);
        // staged on the stack so that the pooled lists get their exact size
        VoxelEntry entries[kLeafSize];
        int32_t candLabels[kLeafSize * kMaxCandidates];
        size_t numEntries = 0, numCand = 0;
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
            CandidateSet c;
            this->addCandidate(c, own);
            for (int f = 0; f < 6; ++f) this->addCandidate(c, nb[f]);
            if (c.n == 0) continue; // only passive label 0: cannot change
            VoxelEntry& e = entries[numEntries++];
            e.offset = uint16_t(it.pos());
            e.n = uint8_t(c.n);
            e.zero = c.zero;
            for (int k = 0; k < c.n; ++k) candLabels[numCand++] = c.label[k];
        }
        if (numEntries == 0) return;
        d.entries.assign(entries, entries + numEntries);
        d.cand.reserve(numCand);
        StencilT stencil(mMls);
        const int32_t* l = candLabels;
        for (size_t k = 0; k < numEntries; ++k) {
            const Coord p = leaf.offsetToGlobalCoord(entries[k].offset);
            for (int j = 0; j < entries[k].n; ++j, ++l) {
                const float s = this->evalSpeed(stencil, p, *l, term);
                d.cand.emplace_back(*l, s);
                d.maxSpeed = std::max(d.maxSpeed, std::abs(s));
            }
        }
        d.any = true;
    });

    Evaluation result;
    for (size_t i = 0; i < leafCount; ++i) {
        result.maxSpeed = std::max(result.maxSpeed, data[i].maxSpeed);
        result.any |= data[i].any;
    }
    mPeakScratchBytes = std::max(mPeakScratchBytes, scratch.memUsage());
    return result;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::motionLimit(
    const Evaluation& e, float dt) const
{
    if (e.any && e.maxSpeed > 0.0f) {
        dt = std::min(dt, float(lsmo_internal::kMaxInterfaceMotionVoxels * mCfl
            * mMls.voxelSize() / e.maxSpeed));
    }
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::kinematicLimit(float dt) const
{
    if (mKinematic.empty()) return dt;
    // kinematic objects move at most kMaxInterfaceMotionVoxels*cfl*dx per step
    const double limit = lsmo_internal::kMaxInterfaceMotionVoxels * mCfl * mMls.voxelSize();
    for (int iter = 0; iter < lsmo_internal::kMaxKinematicShrinkIterations; ++iter) {
        const double d = this->kinematicDisplacement(mTime, mTime + double(dt));
        if (d <= limit) break;
        dt = float(double(dt) * std::min(lsmo_internal::kKinematicStepShrinkMax,
            lsmo_internal::kKinematicStepSafety * limit / d));
    }
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::update(
    float dt, bool any, bool stamp, double stampTime)
{
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();

    // 4. two-smallest projection, written to aux buffer kNext of both
    // trees (initialized with time n) while time n stays in the leaf buffers
    constexpr size_t kNext = PairBuffers::kNext;
    Scratch& scratch = mScratch;
    BoolTreeT& freeze = scratch.freeze;
    std::vector<LeafData>& data = scratch.data;
    tree::LeafManager<const FloatTreeT> leafs(phi);
    const size_t leafCount = leafs.leafCount();
    OPENVDB_ASSERT(leafCount <= data.size());
    std::vector<std::vector<Deferred>>& deferred = scratch.deferred;
    Scratch::resetLists(deferred, mProtected.empty() || !any ? 0 : leafCount);
    if (any) {
        PairBuffers buffers(phi, labels, mGrainSize <= 0);
        OPENVDB_ASSERT(buffers.phi.leafCount() == leafCount);
        // changes involving a protected label are deferred (topology preservation)
        forEachIndex(leafCount, [&](size_t i) {
            const LeafData& d = data[i];
            if (d.entries.empty()) return;
            const auto& leaf = leafs.leaf(i);
            OPENVDB_ASSERT(buffers.phi.leaf(i).origin() == leaf.origin());
            auto& phiBuf = buffers.phiBuffer(i, kNext);
            auto& labBuf = buffers.labelBuffer(i, kNext);
            const auto& labN = buffers.labelBuffer(i, 0); // time n until the swap
            auto* dirtyLeaf = scratch.dirty.probe(leaf.origin());
            OPENVDB_ASSERT(dirtyLeaf);
            const Candidate* cand = d.cand.data();
            for (const VoxelEntry& e : d.entries) {
                const int32_t own = static_cast<int32_t>(labN.getValue(e.offset));
                const float phiN = leaf.getValue(e.offset);
                int32_t l; float v;
                const bool moved = project(e, own, phiN, cand, dt, l, v);
                cand += e.n;
                if (!moved) continue;
                if (this->involvesProtected(own, l)) {
                    deferred[i].push_back(Deferred{leaf.offsetToGlobalCoord(e.offset), own, l, v});
                    continue;
                }
                phiBuf.setValue(e.offset, v);
                labBuf.setValue(e.offset, LabelT(l));
                if (v != phiN || l != own) dirtyLeaf->setOn(e.offset); // re-solve nearby
            }
        });
        mPeakScratchBytes = std::max(mPeakScratchBytes, scratch.memUsage() + buffers.memUsage());

        // 5a. swap both trees together (time n+1 in the leaf buffers); time n is
        // no longer needed
        buffers.swap(kNext);
        buffers.release();
    } // any

    // 5b. deferred (topology-preserving) changes; 5c. kinematic labels at
    // @a stampTime; 5d. orphan cleanup
    if (!mProtected.empty() && any) this->applyDeferred(deferred, phi, labels, freeze);
    if (stamp && !mKinematic.empty()) this->stampKinematic(stampTime, freeze);
    this->cleanOrphans(phi, labels, freeze);

    // 6. band maintenance (topology changes, after the swap)
    this->bandMaintenance(freeze);
    mPeakScratchBytes = std::max(mPeakScratchBytes, scratch.memUsage());
    OPENVDB_ASSERT(mMls.validate());
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::splitStep(
    float maxDt, const Evaluation& h)
{
    OPENVDB_ASSERT(h.any);
    const float dt = this->kinematicLimit(this->motionLimit(h, maxDt));
    const double t1 = mTime + double(dt);
    // parabolic halves: n equal substeps of at most the parabolic stable dt
    const double half = 0.5 * double(dt);
    const double n = std::max(1.0, std::ceil(half / double(this->parabolicStableDt())));
    const float sub = float(half / n);
    bool interrupted = false;
    const double p1 = this->splitPhase(Term::Parabolic, half, sub, /*stampAtEnd=*/false, t1,
        /*interruptible=*/true, /*checkAfterLast=*/true, interrupted);
    if (interrupted) return this->finishInterrupted(float(p1));
    this->splitPhase(Term::Hyperbolic, double(dt), dt, /*stampAtEnd=*/false, t1,
        /*interruptible=*/false, /*checkAfterLast=*/false, interrupted);
    if (util::wasInterrupted(mInterrupter)) return this->finishInterrupted(dt);
    this->splitPhase(Term::Parabolic, half, sub, /*stampAtEnd=*/true, t1,
        /*interruptible=*/true, /*checkAfterLast=*/false, interrupted);
    if (interrupted) return this->finishInterrupted(dt);
    mTime += dt;
    return dt;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
double
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::splitPhase(
    Term term, double duration, float target, bool stampAtEnd, double stampTime,
    bool interruptible, bool checkAfterLast, bool& interrupted)
{
    const double eps = lsmo_internal::kSplitPhaseRelativeEpsilon * duration;
    double remaining = duration;
    while (remaining > eps) {
        const Evaluation e = this->evaluate(term);
        const float dt = this->motionLimit(e, std::min(target, float(remaining)));
        if (!(dt > 0.0f)) break;
        remaining -= double(dt);
        const bool last = !(remaining > eps);
        const bool stamp = stampAtEnd && last && !mKinematic.empty();
        // a sub-update in which nothing moves is skipped
        if ((e.any && e.maxSpeed > 0.0f) || stamp) this->update(dt, e.any, stamp, stampTime);
        if (interruptible && (!last || checkAfterLast) && util::wasInterrupted(mInterrupter)) {
            interrupted = true;
            break;
        }
    }
    return std::max(0.0, duration - remaining);
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
float
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::finishInterrupted(float elapsed)
{
    if (!mKinematic.empty()) {
        // kinematic labels at the time reached; no projection (dt 0)
        this->evaluate(Term::Parabolic);
        this->update(0.0f, /*any=*/false, /*stamp=*/true, mTime + double(elapsed));
    }
    mInterrupted = true;
    mTime += elapsed;
    return elapsed;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::enforcePhiFloor()
{
    if (mProtected.empty()) return;
    const float eps = float(this->phiFloor());
    if (!(eps > 0.0f)) return;
    const LabelTreeT& labels = mMls.labels()->tree();
    tree::LeafManager<FloatTreeT> leafs(mMls.phi()->tree());
    forEachIndex(leafs.leafCount(), [&](size_t i) {
        tree::ValueAccessor<const LabelTreeT> lab(labels);
        auto* dirtyLeaf = mScratch.dirty.probe(leafs.leaf(i).origin());
        OPENVDB_ASSERT(dirtyLeaf);
        for (auto it = leafs.leaf(i).beginValueOn(); it; ++it) {
            if (!(*it < eps)) continue;
            const Coord p = it.getCoord();
            if (this->floorApplies(lab, p, static_cast<int32_t>(lab.getValue(p)))) {
                it.setValue(eps);
                dirtyLeaf->setOn(it.pos());
            }
        }
    });
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
bool
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::validateTopologyFloor() const
{
    if (mProtected.empty()) return true;
    const float eps = float(this->phiFloor());
    tree::ValueAccessor<const LabelTreeT> lab(mMls.labels()->tree());
    for (auto it = mMls.phi()->tree().cbeginValueOn(); it; ++it) {
        if (!(*it < eps)) continue;
        const Coord p = it.getCoord();
        if (this->floorApplies(lab, p, static_cast<int32_t>(lab.getValue(p)))) {
            return false;
        }
    }
    return true;
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::stampKinematic(
    double t, BoolTreeT& freeze)
{
    FloatTreeT& phi = mMls.phi()->tree();
    LabelTreeT& labels = mMls.labels()->tree();
    const math::Transform& xform = mMls.phi()->transform();
    const double dx = mMls.voxelSize();
    const float bg = phi.background();
    // phi of stamped voxels is at least this, so that a voxel on the
    // obstacle surface is not ambiguous (phi = 0)
    const float minPhi = float(std::max(lsmo_internal::kBarrierPhiFraction * dx, this->phiFloor()));
    tree::ValueAccessor<FloatTreeT> phiAcc(phi);
    tree::ValueAccessor<LabelTreeT> labAcc(labels);
    tree::ValueAccessor<BoolTreeT> maskAcc(freeze);
    ChangeLog log(mScratch.dirty);
    // a voxel taken by kinematic @a label with phi @a v
    auto claim = [&](const Coord& p, int32_t label, float v) {
        log.write(phiAcc, labAcc, p, [&] {
            if (v < bg) {
                phiAcc.setValueOn(p, v);
                labAcc.setValueOn(p, LabelT(label));
                maskAcc.setValueOn(p);
            } else {
                phiAcc.setValueOff(p, bg);
                labAcc.setValueOff(p, LabelT(label));
                maskAcc.setValueOff(p);
            }
        });
    };
    std::vector<std::vector<Deferred>> claims(1);
    // the objects at time t, and the object of each queued claim
    struct Object { int32_t label; math::Mat4d inv; const FloatGrid* sdf; };
    std::vector<Object> objects;
    std::unordered_map<Coord, size_t, lsmo_internal::CoordHash> claimObject;
    // labels in increasing order: an object does not take voxels of objects
    // stamped before it
    std::vector<int32_t> kinematicLabels;
    kinematicLabels.reserve(mKinematic.size());
    for (const auto& kv : mKinematic) kinematicLabels.push_back(kv.first);
    std::sort(kinematicLabels.begin(), kinematicLabels.end());
    for (const int32_t label : kinematicLabels) {
        Kinematic& k = mKinematic.at(label);
        if (!k.hasBox) continue;
        const math::Mat4d m = k.motion(t);
        const math::Mat4d inv = m.inverse();
        objects.push_back(Object{label, inv, k.sdf.get()});
        // index box of the moved object, dilated by the band, plus the box
        // stamped last (voxels the object may have left)
        Vec3d lo(std::numeric_limits<double>::max()), hi(-std::numeric_limits<double>::max());
        for (const Vec3d& c : k.corners) {
            const Vec3d p = xform.worldToIndex(m.transform(c));
            lo = math::minComponent(lo, p);
            hi = math::maxComponent(hi, p);
        }
        const int pad = mMls.halfWidth() + 1;
        CoordBBox box(Coord::floor(lo) - Coord(pad), Coord::floor(hi) + Coord(pad + 1));
        const CoordBBox now = box;
        if (!k.lastBox.empty()) box.expand(k.lastBox);
        k.lastBox = now;
        tree::ValueAccessor<const FloatTree> sdfAcc(k.sdf->tree());
        const math::Transform& sdfXform = k.sdf->transform();
        for (auto it = box.begin(); it; ++it) {
            const Coord& p = *it;
            const Vec3d rest = inv.transform(xform.indexToWorld(p));
            const float psiRaw = tools::BoxSampler::sample(sdfAcc, sdfXform.worldToIndex(rest));
            const float psi = psiRaw < 0.0f ? std::min(psiRaw, -minPhi) : std::max(psiRaw, minPhi);
            const int32_t own = static_cast<int32_t>(labAcc.getValue(p));
            // fixed and other kinematic labels are never taken or changed
            if (own != label && this->isFixed(own)) continue;
            if (psi < 0.0f) {
                // inside the obstacle: claims label 0 and unprotected labels;
                // claims on protected labels are decided below
                phiAcc.touchLeaf(p);
                labAcc.touchLeaf(p);
                if (own != label && this->isProtected(own)) {
                    // one claim per voxel when objects overlap
                    if (claimObject.emplace(p, objects.size() - 1).second) {
                        claims[0].push_back(Deferred{p, own, label, -psi});
                    }
                } else {
                    claim(p, label, -psi);
                }
            } else if (own == label) {
                // left by the obstacle: passive label 0, redistanced
                phiAcc.touchLeaf(p);
                labAcc.touchLeaf(p);
                const float v = std::min(psi, bg);
                log.write(phiAcc, labAcc, p, [&] {
                    if (v < bg) {
                        phiAcc.setValueOn(p, v);
                        labAcc.setValueOn(p, LabelT(0));
                    } else {
                        phiAcc.setValueOff(p, bg);
                        labAcc.setValueOff(p, LabelT(0));
                    }
                });
                maskAcc.setValueOff(p);
            } else if (psi < bg) {
                // band just outside: keep the label, distance at most psi
                float cur;
                if (phiAcc.probeValue(p, cur)) {
                    if (psi < cur) {
                        log.write(phiAcc, labAcc, p, [&] { phiAcc.setValueOnly(p, psi); });
                    }
                } else {
                    phiAcc.touchLeaf(p);
                    labAcc.touchLeaf(p);
                    log.write(phiAcc, labAcc, p, [&] {
                        phiAcc.setValueOn(p, psi);
                        labAcc.setValueOn(p, labAcc.getValue(p));
                    });
                }
                if (psi <= float(dx)) maskAcc.setValueOn(p);
            }
        }
    }
    if (claims[0].empty()) return;
    // claims on protected labels: the simple-point test of the colored
    // passes; a rejected voxel keeps its protected label (barrier phi)
    phiAcc.clear();
    labAcc.clear();
    maskAcc.clear();
    std::vector<Deferred> trapped;
    this->coloredPasses(claims, labels,
        [&](const Deferred& c) { claim(c.ijk, c.to, c.phi); },
        [&](const Deferred& c) {
            log.write(phiAcc, labAcc, c.ijk, [&] {
                phiAcc.setValueOn(c.ijk, minPhi);
                labAcc.setValueOn(c.ijk, labAcc.getValue(c.ijk));
            });
            maskAcc.setValueOn(c.ijk);
            trapped.push_back(c);
        });

    // Outward wrap: a protected voxel q left inside an object (trapped) is
    // migrated to the first voxel o outside it along the object's outward
    // normal (psi(o) >= 0.5*dx, neither kinematic nor fixed): o joins the
    // protected label if that is simple, then the object claims q if that
    // is simple; repeated while anything changes, at most kMaxWrapPasses.
    auto psiAt = [&](const Object& o, const Vec3d& world) {
        tree::ValueAccessor<const FloatTree> acc(o.sdf->tree());
        return double(tools::BoxSampler::sample(acc,
            o.sdf->transform().worldToIndex(o.inv.transform(world))));
    };
    for (int pass = 0; pass < lsmo_internal::kMaxWrapPasses && !trapped.empty(); ++pass) {
        phiAcc.clear();
        labAcc.clear();
        maskAcc.clear();
        // targets o of the trapped voxels
        std::vector<std::pair<Deferred, Coord>> moves; // claim of q, target o
        std::unordered_map<Coord, Deferred, lsmo_internal::CoordHash> adds;
        for (const Deferred& c : trapped) {
            const Object& obj = objects[claimObject.at(c.ijk)];
            const Vec3d xq = xform.indexToWorld(c.ijk);
            Vec3d g;
            for (int a = 0; a < 3; ++a) {
                Vec3d e(0.0);
                e[a] = dx;
                g[a] = psiAt(obj, xq + e) - psiAt(obj, xq - e);
            }
            if (!(g.length() > 0.0)) continue;
            g.normalize();
            bool found = false;
            Coord o;
            const int nSteps = int(lsmo_internal::kWrapSearchReachBands * mMls.halfWidth()
                                   / lsmo_internal::kWrapSearchStepVoxels);
            for (int n = 1; n <= nSteps && !found; ++n) {
                const Vec3d x = xq + g * (lsmo_internal::kWrapSearchStepVoxels * dx * n);
                o = Coord::round(xform.worldToIndex(x));
                if (o == c.ijk) continue;
                const int32_t lo = static_cast<int32_t>(labAcc.getValue(o));
                if (this->isFixed(lo)) continue;
                if (psiAt(obj, xform.indexToWorld(o))
                    < lsmo_internal::kObstacleShellInnerVoxels * dx) continue;
                if (lo != c.from && this->isProtected(lo)) continue; // another protected label
                found = true;
            }
            if (found) {
                moves.emplace_back(c, o);
                const int32_t lo = static_cast<int32_t>(labAcc.getValue(o));
                if (lo != c.from && !adds.count(o)) {
                    adds[o] = Deferred{o, lo, c.from,
                                       float(lsmo_internal::kHalfVoxelDistance * dx)};
                }
            } else {
                moves.emplace_back(c, c.ijk);
            }
            // the outer shell near q (kObstacleShellInnerVoxels*dx <= psi <
            // kObstacleShellOuterVoxels*dx) next to the
            // protected label also joins it where simple, so that the label
            // can wrap around the object rather than only reach the target o
            for (auto it = CoordBBox(c.ijk - Coord(2), c.ijk + Coord(2)).begin(); it; ++it) {
                const Coord& p = *it;
                if (adds.count(p)) continue;
                const int32_t lp = static_cast<int32_t>(labAcc.getValue(p));
                if (lp == c.from || this->isFixed(lp) || this->isProtected(lp)) continue;
                const double sp = psiAt(obj, xform.indexToWorld(p));
                if (sp < lsmo_internal::kObstacleShellInnerVoxels * dx
                    || sp >= lsmo_internal::kObstacleShellOuterVoxels * dx) continue;
                bool touches = false;
                for (int f = 0; f < 6 && !touches; ++f) {
                    const Coord q = p + lsmo_internal::faceOffsets()[f];
                    touches = static_cast<int32_t>(labAcc.getValue(q)) == c.from;
                }
                if (touches) {
                    adds[p] = Deferred{p, lp, c.from,
                                       float(lsmo_internal::kHalfVoxelDistance * dx)};
                }
            }
        }
        bool changed = false;
        // 1. o joins the protected label (simple-point test)
        if (!adds.empty()) {
            // in coordinate order (the colored passes apply in list order)
            std::vector<std::vector<Deferred>> addList(1);
            addList[0].reserve(adds.size());
            for (const auto& kv : adds) addList[0].push_back(kv.second);
            std::sort(addList[0].begin(), addList[0].end(),
                [](const Deferred& a, const Deferred& b) { return a.ijk < b.ijk; });
            this->coloredPasses(addList, labels,
                [&](const Deferred& c) {
                    phiAcc.touchLeaf(c.ijk);
                    labAcc.touchLeaf(c.ijk);
                    log.write(phiAcc, labAcc, c.ijk, [&] {
                        phiAcc.setValueOn(c.ijk, c.phi);
                        labAcc.setValueOn(c.ijk, LabelT(c.to));
                    });
                    maskAcc.setValueOn(c.ijk); // at the interface: kept
                    changed = true;
                },
                [](const Deferred&) {});
            phiAcc.clear();
            labAcc.clear();
            maskAcc.clear();
        }
        // 2. the object claims q (simple-point test, after the adds)
        std::vector<std::vector<Deferred>> claimList(1);
        for (const auto& mv : moves) {
            if (mv.second == mv.first.ijk
                || static_cast<int32_t>(labAcc.getValue(mv.second)) == mv.first.from) {
                claimList[0].push_back(mv.first);
            }
        }
        std::unordered_set<Coord, lsmo_internal::CoordHash> claimed;
        this->coloredPasses(claimList, labels,
            [&](const Deferred& c) {
                claim(c.ijk, c.to, c.phi);
                claimed.insert(c.ijk);
                changed = true;
            },
            [](const Deferred&) {});
        std::vector<Deferred> left;
        for (const Deferred& c : trapped) if (!claimed.count(c.ijk)) left.push_back(c);
        trapped.swap(left);
        if (!changed) break;
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::cleanOrphans(
    FloatTreeT& phi, LabelTreeT& labels, BoolTreeT& freeze)
{
    tree::LeafManager<const FloatTreeT> leafs(phi);
    std::vector<std::vector<Fix>> fixes(leafs.leafCount());
    forEachIndex(leafs.leafCount(), [&](size_t i) {
        tree::ValueAccessor<const FloatTreeT> phiAcc(phi);
        tree::ValueAccessor<const LabelTreeT> labAcc(labels);
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
    mPeakScratchBytes = std::max(mPeakScratchBytes, mScratch.memUsage() + vectorBytes(fixes));
    tree::ValueAccessor<FloatTreeT> phiAcc(phi);
    tree::ValueAccessor<LabelTreeT> labAcc(labels);
    tree::ValueAccessor<BoolTreeT> maskAcc(freeze);
    ChangeLog log(mScratch.dirty);
    for (const auto& list : fixes) {
        for (const Fix& fx : list) {
            // protected labels: only simple changes, tested on the
            // labels as updated by the fixes applied so far (serial order)
            const int32_t own = static_cast<int32_t>(labAcc.getValue(fx.ijk));
            const int32_t to = static_cast<int32_t>(fx.label);
            if (this->involvesProtected(own, to)
                && !this->topologyAllows(labAcc, fx.ijk, own, to)) continue;
            log.write(phiAcc, labAcc, fx.ijk, [&] {
                phiAcc.setValueOnly(fx.ijk, fx.phi);
                labAcc.setValueOnly(fx.ijk, fx.label);
            });
            maskAcc.setValueOff(fx.ijk); // flagged for redistancing
        }
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
template<typename AcceptT, typename RejectT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::coloredPasses(
    const std::vector<std::vector<Deferred>>& deferred, const LabelTreeT& labels,
    const AcceptT& accept, const RejectT& reject) const
{
    // 27 colored passes (coordinates mod 3): changes of one color have
    // disjoint 26-neighborhoods, so they are decided independently (in
    // parallel) on the labels as updated by the earlier passes, then
    // applied. The result does not depend on the thread count.
    std::vector<Deferred> byColor[27];
    for (const auto& list : deferred) {
        for (const Deferred& c : list) {
            auto mod3 = [](int v) { return ((v % 3) + 3) % 3; };
            byColor[mod3(c.ijk.x()) + 3 * mod3(c.ijk.y()) + 9 * mod3(c.ijk.z())].push_back(c);
        }
    }
    for (const std::vector<Deferred>& pass : byColor) {
        if (pass.empty()) continue;
        std::vector<char> ok(pass.size(), 0);
        forEachIndex(pass.size(), [&](size_t i) {
            tree::ValueAccessor<const LabelTreeT> lab(labels);
            const Deferred& c = pass[i];
            ok[i] = this->topologyAllows(lab, c.ijk, c.from, c.to);
        });
        for (size_t i = 0; i < pass.size(); ++i) {
            if (ok[i]) accept(pass[i]);
            else reject(pass[i]);
        }
    }
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::applyDeferred(
    const std::vector<std::vector<Deferred>>& deferred,
    FloatTreeT& phi, LabelTreeT& labels, BoolTreeT& freeze)
{
    const float barrier = float(std::max(lsmo_internal::kBarrierPhiFraction * mMls.voxelSize(),
                                         this->phiFloor()));
    tree::ValueAccessor<FloatTreeT> phiAcc(phi);
    tree::ValueAccessor<LabelTreeT> labAcc(labels);
    tree::ValueAccessor<BoolTreeT> maskAcc(freeze);
    ChangeLog log(mScratch.dirty);
    this->coloredPasses(deferred, labels,
        [&](const Deferred& c) {
            log.write(phiAcc, labAcc, c.ijk, [&] {
                phiAcc.setValueOnly(c.ijk, c.phi);
                labAcc.setValueOnly(c.ijk, LabelT(c.to));
            });
        },
        [&](const Deferred& c) {
            // rejected: keep the label, barrier just inside its region
            log.write(phiAcc, labAcc, c.ijk, [&] { phiAcc.setValueOnly(c.ijk, barrier); });
            maskAcc.setValueOn(c.ijk);
        });
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

    // b. activate the inactive face neighbors of voxels with
    // phi < (halfWidth-1)*dx; label of the smallest-phi active face neighbor
    // (ties to the smaller label). Candidates are collected per leaf in
    // parallel on the unmodified trees, then applied serially in leaf order;
    // the result is the (phi, label) minimum per voxel, so it does not
    // depend on the order or the thread count.
    {
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
        mPeakScratchBytes = std::max(mPeakScratchBytes, mScratch.memUsage() + vectorBytes(found));
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
            mScratch.dirty.setOn(kv.first); // newly active
            phiW.setValueOnly(kv.first, bg);
            // an activated voxel keeps its inactive label when the change
            // involves a protected label (topology preservation) or would
            // make it a fixed or kinematic label's voxel (they never grow)
            const LabelT l = labW.getValue(kv.first);
            if (this->isFixed(static_cast<int32_t>(l))
                || this->involvesProtected(static_cast<int32_t>(kv.second),
                    static_cast<int32_t>(l))) {
                labW.setValueOnly(kv.first, kv.second);
            }
        }
    }

    // c. redistance: projected and fixed voxels frozen, the rest solved with
    // signed reflection to the tolerance (the phi floor of protected
    // interfaces holds for its input and its output). Incrementally, only
    // the leaves near a change since the last redistance are re-solved.
    Scratch& sc = mScratch;
    sc.dirty.touch(phi);
    this->enforcePhiFloor();
    {
        MultiLevelSetRedistance<MultiLevelSetT> redist(mMls);
        redist.setTolerance(mTolerance);
        redist.setGrainSize(mGrainSize);
        if (mFullRedistance || !mPrimed) {
            mRedistanceIterations = redist.redistance(freeze);
        } else {
#ifdef OPENVDB_ENABLE_ASSERTS
            this->checkChangesRecorded();
#endif
            mRedistanceIterations = this->incrementalRedistance(redist, freeze);
        }
        if (!redist.converged()) ++mRedistanceCapHits;
        this->rememberSeeds(freeze);
        mPrimed = true;
    }
#ifdef OPENVDB_ENABLE_ASSERTS
    mCheckPhi.reset(new FloatTreeT(phi));
    mCheckLabels.reset(new LabelTreeT(labels));
#endif
    // Changes from here on are re-solved at the next step. The values the
    // redistance just wrote are its output, not changes: they are left
    // unmarked (marking them would make the whole band dirty every step).
    sc.dirty.reset(phi);
    this->enforcePhiFloor();

    // d. deactivate phi >= halfWidth*dx in both trees (labels kept)
    {
        tree::LeafManager<FloatTreeT> leafs(phi);
        forEachIndex(leafs.leafCount(), [&](size_t i) {
            auto& leaf = leafs.leaf(i);
            auto* labLeaf = labels.probeLeaf(leaf.origin());
            auto* dirtyLeaf = sc.dirty.probe(leaf.origin());
            OPENVDB_ASSERT(dirtyLeaf);
            for (auto it = leaf.beginValueOn(); it; ++it) {
                if (*it < bg) continue;
                dirtyLeaf->setOn(it.pos());
                leaf.setValueOff(it.pos(), bg);
                if (labLeaf) labLeaf->setActiveState(it.pos(), false);
            }
        });
    }
    mMls.pruneInactive();
}


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
int
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::incrementalRedistance(
    MultiLevelSetRedistance<MultiLevelSetT>& redist, const BoolTreeT& freeze)
{
    using PhiLeafT = typename FloatTreeT::LeafNodeType;
    using MaskT = typename BoolLeafT::NodeMaskType;
    constexpr int kDim = int(PhiLeafT::DIM);
    FloatTreeT& phi = mMls.phi()->tree();
    Scratch& sc = mScratch;
    tree::LeafManager<FloatTreeT> leafs(phi);
    const size_t n = leafs.leafCount();

    // 1. Voxels that became or stopped being frozen since the last
    // redistance change the Dirichlet data too. The masks have an entry at
    // every phi leaf.
    forEachIndex(n, [&](size_t i) {
        const PhiLeafT& leaf = leafs.leaf(i);
        const BoolLeafT* f = freeze.probeConstLeaf(leaf.origin());
        const MaskT* prev = static_cast<const LeafMasks&>(sc.prevSeeds).probe(leaf.origin());
        const MaskT now = f ? f->getValueMask() & leaf.getValueMask() : MaskT();
        MaskT* dirtyLeaf = sc.dirty.probe(leaf.origin());
        OPENVDB_ASSERT(dirtyLeaf);
        *dirtyLeaf |= prev ? now ^ *prev : now;
    });

    // 2. Solve every active, non-frozen voxel of the leaves within
    // halfWidth + 1 voxels of a dirty voxel (taken at leaf granularity, a
    // superset). The region follows the Eikonal domain of dependence (a
    // band value below halfWidth*dx, plus one voxel for the second-order
    // stencil) but is a heuristic: the monotone second-order update makes
    // converged values depend on the order voxels are reached, and this
    // solve's stopping test is local to the solved leaves. It matches the
    // full solve to within the tolerance in the tested scenes for
    // halfWidth <= 3 only (see the class documentation).
    const int reach = (mMls.halfWidth() + 1 + kDim - 1) / kDim; // in leaves
    std::vector<char> solveLeaf(n, 0);
    forEachIndex(n, [&](size_t i) {
        const Coord o = leafs.leaf(i).origin();
        const LeafMasks& dirtyMasks = sc.dirty;
        bool dirty = false;
        for (int a = -reach; a <= reach && !dirty; ++a) {
            for (int b = -reach; b <= reach && !dirty; ++b) {
                for (int c = -reach; c <= reach && !dirty; ++c) {
                    const MaskT* d = dirtyMasks.probe(o.offsetBy(a * kDim, b * kDim, c * kDim));
                    dirty = d && !d->isOff();
                }
            }
        }
        solveLeaf[i] = dirty;
    });
    std::vector<PhiLeafT*> leaves;
    std::vector<MaskT> masks;
    {
        tree::ValueAccessor<const BoolTreeT> freezeAcc(freeze);
        for (size_t i = 0; i < n; ++i) {
            if (!solveLeaf[i]) continue;
            PhiLeafT& leaf = leafs.leaf(i);
            MaskT m = leaf.getValueMask();
            if (const BoolLeafT* f = freezeAcc.probeConstLeaf(leaf.origin())) {
                m -= f->getValueMask();
            } else if (freezeAcc.isValueOn(leaf.origin())) {
                continue; // frozen tile
            }
            if (m.isOff()) continue;
            leaves.push_back(&leaf);
            masks.push_back(m);
        }
    }
    mPeakScratchBytes = std::max(mPeakScratchBytes, sc.memUsage()
        + vectorBytes(solveLeaf) + vectorBytes(leaves) + vectorBytes(masks));
    return redist.redistance(freeze, leaves, masks);
}


#ifdef OPENVDB_ENABLE_ASSERTS
template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::checkChangesRecorded() const
{
    if (!mCheckPhi || !mCheckLabels) return;
    const FloatTreeT& phi = mMls.phi()->tree();
    const LabelTreeT& labels = mMls.labels()->tree();
    tree::ValueAccessor<const FloatTreeT> phiNow(phi), phiThen(*mCheckPhi);
    tree::ValueAccessor<const LabelTreeT> labNow(labels), labThen(*mCheckLabels);
    auto check = [&](const Coord& p) {
        float a = 0.0f, b = 0.0f;
        const bool onA = phiThen.probeValue(p, a), onB = phiNow.probeValue(p, b);
        if (onA == onB && (!onA || a == b) && labThen.getValue(p) == labNow.getValue(p)) return;
        const auto* m = mScratch.dirty.probe(LeafMasks::originOf(p));
        OPENVDB_ASSERT_MESSAGE(m && m->isOn(BoolLeafT::coordToOffset(p)),
            "incremental redistance: a changed voxel is not marked dirty");
    };
    auto visit = [&](const auto& t) {
        for (auto leaf = t.cbeginLeaf(); leaf; ++leaf) {
            for (Index n = 0; n < BoolLeafT::SIZE; ++n) check(leaf->offsetToGlobalCoord(n));
        }
    };
    visit(phi);
    visit(*mCheckPhi);
    visit(labels);
    visit(*mCheckLabels);
}
#endif


template<typename MultiLevelSetT, typename SpeedT, typename InterruptT>
void
LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>::rememberSeeds(const BoolTreeT& freeze)
{
    const FloatTreeT& phi = mMls.phi()->tree();
    tree::LeafManager<const FloatTreeT> leafs(phi);
    mScratch.prevSeeds.reset(phi);
    forEachIndex(leafs.leafCount(), [&](size_t i) {
        const auto& leaf = leafs.leaf(i);
        const BoolLeafT* f = freeze.probeConstLeaf(leaf.origin());
        if (!f) return;
        auto* prev = mScratch.prevSeeds.probe(leaf.origin());
        OPENVDB_ASSERT(prev);
        *prev = f->getValueMask() & leaf.getValueMask();
    });
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_HAS_BEEN_INCLUDED
