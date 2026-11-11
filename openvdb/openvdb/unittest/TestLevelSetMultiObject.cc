// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTubes.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VelocityFields.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "LevelSetMultiObjectFixtures.h"
#include "LevelSetMultiObjectTopology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class TestLevelSetMultiObject: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

namespace {

using namespace openvdb;
using MLS = lsmo_test::MLS;
using LabelTree = lsmo_test::LabelTree;
using lsmo_test::contactPairs;
using lsmo_test::junctionSet;
using lsmo_test::labelVoxelCounts;
using lsmo_test::makeTwoSpheres;
using lsmo_test::makeVelocity;
using lsmo_test::Vortex;
using lsmo_test::TripleJunction;

// 16^3 block: label 1 for x<8, 2 for x>=8, phi = |x - 7.5|*dx clamped to 3*dx.
template<typename MultiLevelSetT = MLS>
typename MultiLevelSetT::Ptr
makePair(double dx = 1.0)
{
    using LabelGridT = typename MultiLevelSetT::LabelGridT;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3.0 * dx));
    typename LabelGridT::Ptr lab = LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const Coord c(i, j, k);
                const float d = std::min(float(std::abs(i - 7.5) * dx), float(3.0 * dx));
                pa.setValue(c, d);
                la.setValue(c, i < 8 ? 1 : 2);
            }
    return typename MultiLevelSetT::Ptr(new MultiLevelSetT(phi, lab, 3));
}

} // anonymous namespace

namespace {
const char* const kBackgroundMessage =
    "phi background must equal halfWidth * voxel size and the label background must be 0";

template<typename T, typename = void>
struct HasSetClampSpeed : std::false_type {};
template<typename T>
struct HasSetClampSpeed<T, decltype(void(std::declval<T&>().setClampSpeed(true)))>
    : std::true_type {};
template<typename T, typename = void>
struct HasClampSpeed : std::false_type {};
template<typename T>
struct HasClampSpeed<T, decltype(void(std::declval<const T&>().clampSpeed()))>
    : std::true_type {};

struct DetectionSpeed
{
    float operator()(const Coord&, int32_t, const tools::LabelAwareStencil<MLS>&) const
    { return 0.0f; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return 1.0f; }
};
} // anonymous namespace

TEST_F(TestLevelSetMultiObject, NoClampSpeedOption)
{
    using Evolver = lsmo_frames::FrameEvolver<MLS, DetectionSpeed>;
    static_assert(!HasSetClampSpeed<Evolver>::value, "setClampSpeed must not exist");
    static_assert(!HasClampSpeed<Evolver>::value, "clampSpeed must not exist");
    SUCCEED();
}

TEST_F(TestLevelSetMultiObject, HolderValidate)
{
    auto m = makePair();
    std::string msg;
    EXPECT_TRUE(m->validate(&msg));
    EXPECT_EQ(m->phi()->metaValue<std::string>("multi_level_set_role"), "phi");
    EXPECT_EQ(m->labels()->metaValue<std::string>("multi_level_set_role"), "labels");
    EXPECT_EQ(m->phi()->getGridClass(), GRID_UNKNOWN);

    m->phi()->tree().setValueOn(Coord(2, 2, 2), -1.0f);
    msg.clear();
    EXPECT_FALSE(m->validate(&msg));
    EXPECT_EQ(msg, "phi must be non-negative and finite at every active voxel");

    auto m2 = makePair();
    m2->labels()->tree().setValueOn(Coord(40, 0, 0), 1);
    msg.clear();
    EXPECT_FALSE(m2->validate(&msg));
    EXPECT_EQ(msg, "phi and label trees have different active topology");
}

TEST_F(TestLevelSetMultiObject, HolderThrows)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(math::Transform::createLinearTransform(1.0));
    lab->setTransform(math::Transform::createLinearTransform(0.5));
    EXPECT_THROW(MLS(phi, lab, 3), ValueError);

    lab->setTransform(math::Transform::createLinearTransform(1.0));
    EXPECT_THROW(MLS(phi, lab, 1), ValueError);
    // the phi background must be halfWidth*dx
    FloatGrid::Ptr phiHw2 = FloatGrid::create(2.0f);
    phiHw2->setTransform(math::Transform::createLinearTransform(1.0));
    EXPECT_NO_THROW(MLS(phiHw2, lab, 2));

    auto nonUniform = math::Transform::Ptr(new math::Transform(math::MapBase::Ptr(new math::ScaleMap(Vec3d(1.0, 2.0, 1.0)))));
    FloatGrid::Ptr phi2 = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab2 = MLS::LabelGridT::create(0);
    phi2->setTransform(nonUniform);
    lab2->setTransform(nonUniform->copy());
    EXPECT_THROW(MLS(phi2, lab2, 3), ValueError);
}

TEST_F(TestLevelSetMultiObject, HolderInvariantThrows)
{
    auto pair = [](float bg, int32_t labelBg, double dx = 0.5) {
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(bg);
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(static_cast<MLS::LabelT>(labelBg));
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        phi->tree().setValueOn(Coord(0, 0, 0), 0.25f);
        lab->tree().setValueOn(Coord(0, 0, 0), 1);
        return std::make_pair(phi, lab);
    };
    // phi background must be halfWidth*dx = 1.5 (relative tolerance 1e-6)
    {
        auto g = pair(1.5f, 0);
        EXPECT_NO_THROW(MLS(g.first, g.second, 3));
        EXPECT_THROW(MLS(g.first, g.second, 4), ValueError);
        g = pair(1.5f * (1.0f + 1e-7f), 0);
        EXPECT_NO_THROW(MLS(g.first, g.second, 3));
        g = pair(1.5f * (1.0f + 1e-5f), 0);
        EXPECT_THROW(MLS(g.first, g.second, 3), ValueError);
        g = pair(3.0f, 0); // halfWidth in voxels, not world units
        EXPECT_THROW(MLS(g.first, g.second, 3), ValueError);
    }
    // label background must be 0
    {
        auto g = pair(1.5f, 1);
        EXPECT_THROW(MLS(g.first, g.second, 3), ValueError);
    }
    // phi must be finite and non-negative at active voxels
    for (const float bad : {-0.25f, std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity()}) {
        auto g = pair(1.5f, 0);
        g.first->tree().setValueOn(Coord(1, 0, 0), bad);
        g.second->tree().setValueOn(Coord(1, 0, 0), 1);
        EXPECT_THROW(MLS(g.first, g.second, 3), ValueError) << bad;
    }
    // inactive values are not checked
    {
        auto g = pair(1.5f, 0);
        g.first->tree().setValueOff(Coord(2, 0, 0), -1.0f);
        EXPECT_NO_THROW(MLS(g.first, g.second, 3));
    }
    // validate() also checks the backgrounds
    {
        auto g = pair(1.5f, 0);
        MLS m(g.first, g.second, 3);
        std::string msg;
        EXPECT_TRUE(m.validate(&msg)) << msg;
        m.phi()->tree().root().setBackground(2.0f, /*updateChildNodes=*/true);
        EXPECT_FALSE(m.validate(&msg));
        EXPECT_EQ(msg, kBackgroundMessage);
        m.phi()->tree().root().setBackground(1.5f, /*updateChildNodes=*/true);
        EXPECT_TRUE(m.validate(&msg)) << msg;
        m.labels()->tree().root().setBackground(3, /*updateChildNodes=*/true);
        msg.clear();
        EXPECT_FALSE(m.validate(&msg));
        EXPECT_EQ(msg, kBackgroundMessage);
    }
}

namespace {

// Bitwise equality of two trees: active topology, leaf layout and every
// value (active or not).
template<typename TreeT>
Index64
treeDifferences(const TreeT& a, const TreeT& b)
{
    if (!a.hasSameTopology(b) || a.leafCount() != b.leafCount()) return Index64(1) << 40;
    Index64 diff = 0;
    tree::ValueAccessor<const TreeT> acc(b);
    for (auto it = a.cbeginValueAll(); it; ++it) {
        const auto va = *it;
        const auto vb = acc.getValue(it.getCoord());
        if (std::memcmp(&va, &vb, sizeof(va)) != 0) ++diff;
    }
    return diff;
}

// Labels 1..3 in 8^3 blocks over [0, 23]^3, label 0 outside.
template<typename GridT>
typename GridT::Ptr
makeBlockLabels()
{
    typename GridT::Ptr labels = GridT::create(0);
    auto acc = labels->getAccessor();
    for (int i = 0; i < 24; ++i)
        for (int j = 0; j < 24; ++j)
            for (int k = 0; k < 24; ++k) {
                acc.setValue(Coord(i, j, k),
                    typename GridT::ValueType(1 + ((i / 8 + j / 8 + k / 8) % 3)));
            }
    return labels;
}

// fromLabelGrid, a few evolver steps (1.5 x the time of five full steps) and
// extractSignedDistance with label tree type LabelTreeT; returns the final phi
// tree and label-1 distance.
template<typename LabelTreeT>
std::pair<FloatTree::Ptr, FloatGrid::Ptr>
runWithLabelTree()
{
    using MLST = tools::MultiLevelSet<FloatTree, LabelTreeT>;
    using ForcesT = tools::MultiObjectForces<MLST>;
    auto labels = makeBlockLabels<Grid<LabelTreeT>>();
    auto m = MLST::fromLabelGrid(*labels);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));
    ForcesT forces;
    forces.setPressure(FloatGrid::create(1.0f), 1.0f);
    forces.setCurvatureWeight(0.1f);
    forces.setFixedLabels({3});
    lsmo_frames::FrameEvolver<MLST, ForcesT> lsmo(*m, forces);
    // 1.5 x the time of five full steps
    double t5 = 0.0;
    for (int n = 0; n < 5; ++n) t5 += lsmo.step(1.0f);
    for (double t = t5; t < 1.5 * t5 - 1e-9;) t += lsmo.step(float(1.5 * t5 - t));
    EXPECT_TRUE(m->validate(&msg)) << msg;
    tools::MultiLevelSetRedistance<MLST>(*m).redistance();
    return std::make_pair(FloatTree::Ptr(new FloatTree(m->phi()->tree())),
        m->extractSignedDistance(1));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, OtherLabelTreeTypes)
{
    // The distance tree must hold floats (a static_assert); the label tree
    // may hold any integer type. Int16 (the default), Int64 and UInt32 labels
    // give results identical to Int32 labels.
    const auto ref = runWithLabelTree<Int32Tree>();
    const auto i16 = runWithLabelTree<tools::Int16LabelTree>();
    const auto i64 = runWithLabelTree<Int64Tree>();
    const auto u32 = runWithLabelTree<UInt32Tree>();
    ASSERT_GT(ref.second->activeVoxelCount(), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *i16.first), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), i16.second->tree()), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *i64.first), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *u32.first), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), i64.second->tree()), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), u32.second->tree()), Index64(0));
}

TEST_F(TestLevelSetMultiObject, Int16LabelsDefault)
{
    // The default label tree holds 16-bit labels: half the label bytes of
    // the Int32 variant for the same pair.
    static_assert(std::is_same<MLS::LabelGridT::TreeType, tools::Int16LabelTree>::value,
        "the default label tree is Int16LabelTree");
    static_assert(std::is_same<tools::Int16LabelGrid, Grid<tools::Int16LabelTree>>::value,
        "Int16LabelGrid is the grid of Int16LabelTree");
    using MLS32 = tools::MultiLevelSet<FloatTree, Int32Tree>;
    auto m16 = makeTwoSpheres(0.1);
    auto m32 = makeTwoSpheres<MLS32>(0.1);
    std::cout << "Int16LabelsDefault: two-sphere labels memUsage int16 "
              << m16->labels()->memUsage() << " B, int32 " << m32->labels()->memUsage()
              << " B" << std::endl;
    EXPECT_LT(m16->labels()->memUsage(), m32->labels()->memUsage());
    // same pair, same labels
    EXPECT_EQ(treeDifferences(m16->phi()->tree(), m32->phi()->tree()), Index64(0));
    EXPECT_TRUE(m16->labels()->tree().hasSameTopology(m32->labels()->tree()));
    tree::ValueAccessor<const Int32Tree> acc32(m32->labels()->tree());
    for (auto it = m16->labels()->tree().cbeginValueAll(); it; ++it) {
        ASSERT_EQ(int32_t(*it), acc32.getValue(it.getCoord())) << it.getCoord();
    }
}

TEST_F(TestLevelSetMultiObject, Int16RangeThrows)
{
    // Labels are int32_t in the API; storing one outside the range of the
    // 16-bit label tree throws instead of truncating.
    constexpr int32_t kTooLarge = 40000, kTooSmall = -32769;
    constexpr int32_t kMinInt16 = std::numeric_limits<int16_t>::min();
    constexpr int32_t kMaxInt16 = std::numeric_limits<int16_t>::max();
    auto blockLabels = [](int32_t a, int32_t b) {
        Int32Grid::Ptr g = Int32Grid::create(0);
        auto acc = g->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(7)).begin(); p; ++p) {
            acc.setValue(*p, (*p)[0] < 4 ? a : b);
        }
        return g;
    };
    // fromLabelGrid, active voxel out of range (both directions)
    EXPECT_THROW(MLS::fromLabelGrid(*blockLabels(1, kTooLarge)), ValueError);
    EXPECT_THROW(MLS::fromLabelGrid(*blockLabels(kTooSmall, 1)), ValueError);
    // ... an inactive voxel and an inactive tile out of range
    {
        Int32Grid::Ptr g = blockLabels(1, 2);
        g->tree().setValueOff(Coord(100, 0, 0), kTooLarge);
        EXPECT_THROW(MLS::fromLabelGrid(*g), ValueError);
        Int32Grid::Ptr t = blockLabels(1, 2);
        t->tree().addTile(1, Coord(256, 0, 0), kTooLarge, false);
        EXPECT_THROW(MLS::fromLabelGrid(*t), ValueError);
    }
    // the extremes of the range are accepted and kept exactly
    {
        auto m = MLS::fromLabelGrid(*blockLabels(kMinInt16, kMaxInt16));
        std::string msg;
        EXPECT_TRUE(m->validate(&msg)) << msg;
        EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{kMinInt16, kMaxInt16}));
    }
    // the Int32 variant keeps large labels
    {
        using MLS32 = tools::MultiLevelSet<FloatTree, Int32Tree>;
        auto m = MLS32::fromLabelGrid(*blockLabels(1, kTooLarge));
        EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, kTooLarge}));
    }
    // fromSignedGrids, a key out of range
    {
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(-6, 0, 0), 1.0f);
        in[kTooLarge] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(6, 0, 0), 1.0f);
        EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    }
    // the evolver's kinematic label is stored into the label tree
    {
        auto m = makeTwoSpheres(1.0);
        lsmo_test::Forces forces;
        lsmo_frames::FrameEvolver<MLS, lsmo_test::Forces> lsmo(*m, forces);
        FloatGrid::Ptr sdf = tools::createLevelSetSphere<FloatGrid>(3.0f, Vec3f(0), 1.0f);
        auto still = [](double) { return math::Mat4d::identity(); };
        EXPECT_THROW(lsmo.setKinematicLabel(kTooLarge, sdf, still), ValueError);
        EXPECT_NO_THROW(lsmo.setKinematicLabel(kMaxInt16, sdf, still));
    }
    // reading a label the tree cannot hold finds nothing (no aliasing:
    // 65537 truncates to 1 in 16 bits)
    {
        auto m = makeTwoSpheres(1.0);
        constexpr int32_t kAliasOfOne = 65537;
        EXPECT_EQ(m->extractSignedDistance(kAliasOfOne)->activeVoxelCount(), Index64(0));
        EXPECT_GT(m->extractSignedDistance(1)->activeVoxelCount(), Index64(0));
    }
}

TEST_F(TestLevelSetMultiObject, Int16GridIO)
{
    // OpenVDB does not register int16 grids; registerMultiLevelSetGridTypes()
    // does (idempotent), so the pair round-trips through a .vdb file.
    tools::registerMultiLevelSetGridTypes();
    tools::registerMultiLevelSetGridTypes();
    EXPECT_TRUE(tools::Int16LabelGrid::isRegistered());
    auto m = makeTwoSpheres(0.25);
    m->phi()->setName("phi");
    m->labels()->setName("labels");
    lsmo_test::ScopedDebugDir dir;
    const std::string path = (dir.path() / "int16_pair.vdb").string();
    {
        io::File file(path);
        file.write(GridPtrVec{m->phi(), m->labels()});
        file.close();
    }
    io::File file(path);
    file.open();
    FloatGrid::Ptr phi = gridPtrCast<FloatGrid>(file.readGrid("phi"));
    tools::Int16LabelGrid::Ptr labels =
        gridPtrCast<tools::Int16LabelGrid>(file.readGrid("labels"));
    file.close();
    ASSERT_TRUE(phi);
    ASSERT_TRUE(labels);
    MLS back(phi, labels, m->halfWidth());
    std::string msg;
    EXPECT_TRUE(back.validate(&msg)) << msg;
    EXPECT_EQ(treeDifferences(m->phi()->tree(), phi->tree()), Index64(0));
    EXPECT_TRUE(labels->tree().hasSameTopology(m->labels()->tree()));
    tree::ValueAccessor<const tools::Int16LabelTree> acc(m->labels()->tree());
    for (auto it = labels->tree().cbeginValueAll(); it; ++it) {
        ASSERT_EQ(*it, acc.getValue(it.getCoord())) << it.getCoord();
    }
    EXPECT_EQ(back.labelSet(), m->labelSet());
}

TEST_F(TestLevelSetMultiObject, FactoriesSerialMatchParallel)
{
    // fromLabelGrid, fromSignedGrids and extractSignedDistance give
    // bit-identical results for every grain size (<= 0: serial).
    Int32Grid::Ptr labels = makeBlockLabels<Int32Grid>();
    auto serial = MLS::fromLabelGrid(*labels, 3, 0);
    for (const int grain : {1, 4}) {
        auto parallel = MLS::fromLabelGrid(*labels, 3, grain);
        EXPECT_EQ(treeDifferences(serial->phi()->tree(), parallel->phi()->tree()), Index64(0));
        EXPECT_EQ(treeDifferences(serial->labels()->tree(), parallel->labels()->tree()),
            Index64(0));
    }

    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(9.0f, Vec3f(-6.0f, 0.3f, 0.1f), 0.5f);
    in[2] = tools::createLevelSetSphere<FloatGrid>(8.0f, Vec3f(7.5f, 0.2f, -0.4f), 0.5f);
    in[3] = tools::createLevelSetSphere<FloatGrid>(6.0f, Vec3f(0.5f, 11.0f, 0.3f), 0.5f);
    auto s2 = MLS::fromSignedGrids(in, 3, -1);
    for (const int grain : {1, 4}) {
        auto p2 = MLS::fromSignedGrids(in, 3, grain);
        EXPECT_EQ(treeDifferences(s2->phi()->tree(), p2->phi()->tree()), Index64(0));
        EXPECT_EQ(treeDifferences(s2->labels()->tree(), p2->labels()->tree()), Index64(0));
    }

    for (const MLS::Ptr& m : {serial, s2}) {
        for (const int32_t l : m->labelSet()) {
            for (const int hw : {-1, 2, 5}) {
                FloatGrid::Ptr a = m->extractSignedDistance(l, hw, 0);
                FloatGrid::Ptr b = m->extractSignedDistance(l, hw, 1);
                FloatGrid::Ptr c = m->extractSignedDistance(l, hw, 3);
                ASSERT_GT(a->activeVoxelCount(), Index64(0));
                EXPECT_EQ(treeDifferences(a->tree(), b->tree()), Index64(0)) << l << " " << hw;
                EXPECT_EQ(treeDifferences(a->tree(), c->tree()), Index64(0)) << l << " " << hw;
            }
        }
    }
}

TEST_F(TestLevelSetMultiObject, ExtractInterrupt)
{
    struct Interrupter: public util::NullInterrupter
    {
        int calls = 0, starts = 0, ends = 0, fireAt = 3;
        void start(const char* = nullptr) override { ++starts; }
        void end() override { ++ends; }
        bool wasInterrupted(int = -1) override { return ++calls >= fireAt; }
    };
    auto m = makeTwoSpheres(1.0);
    FloatGrid::Ptr full = m->extractSignedDistance(1);

    // never fires: identical result
    Interrupter quiet;
    quiet.fireAt = std::numeric_limits<int>::max();
    FloatGrid::Ptr same = m->extractSignedDistance(1, -1, 1, &quiet);
    EXPECT_GT(quiet.calls, 0);
    EXPECT_EQ(quiet.starts, 1);
    EXPECT_EQ(quiet.ends, 1);
    EXPECT_EQ(treeDifferences(full->tree(), same->tree()), Index64(0));

    // fires after two Jacobi iterations: still a valid level set grid
    Interrupter early;
    FloatGrid::Ptr part = m->extractSignedDistance(1, -1, 1, &early);
    EXPECT_EQ(early.calls, early.fireAt);
    EXPECT_EQ(early.ends, 1);
    EXPECT_EQ(part->getGridClass(), GRID_LEVEL_SET);
    EXPECT_FLOAT_EQ(part->background(), full->background());
    const float bg = part->background();
    for (auto it = part->tree().cbeginValueAll(); it; ++it) {
        ASSERT_TRUE(std::isfinite(*it));
        if (it.isValueOn()) ASSERT_LT(std::abs(*it), bg);
        else ASSERT_EQ(std::abs(*it), bg);
    }
    EXPECT_GT(part->activeVoxelCount(), Index64(0));
    EXPECT_LT(part->activeVoxelCount(), full->activeVoxelCount());
}

TEST_F(TestLevelSetMultiObject, HolderSignedDistance)
{
    auto m = makePair(0.5);
    const Coord c(3, 0, 0);
    const float d = m->phi()->tree().getValue(c);
    EXPECT_GT(d, 0.0f);
    EXPECT_FLOAT_EQ(m->signedDistance(c, 1), -d);
    EXPECT_FLOAT_EQ(m->signedDistance(c, 2), d);
    EXPECT_DOUBLE_EQ(m->voxelSize(), 0.5);
    EXPECT_EQ(m->halfWidth(), 3);
}

TEST_F(TestLevelSetMultiObject, HolderLabelSet)
{
    // labels beyond 16 bits: the Int32 variant
    auto m = makePair<tools::MultiLevelSet<FloatTree, Int32Tree>>();
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    m->labels()->tree().setValueOn(Coord(0, 0, 0), -7);
    m->labels()->tree().setValueOn(Coord(1, 0, 0), 1 << 30);
    m->labels()->tree().setValueOn(Coord(2, 0, 0), 0);
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{-7, 1, 2, 1 << 30}));
}

TEST_F(TestLevelSetMultiObject, HolderDilateErode)
{
    auto m = makePair();
    const Index64 n0 = m->phi()->activeVoxelCount();
    m->dilate(2);
    EXPECT_GT(m->phi()->activeVoxelCount(), n0);
    EXPECT_EQ(m->phi()->activeVoxelCount(), m->labels()->activeVoxelCount());
    EXPECT_TRUE(m->validate());
    // new voxels outside x<0 copy label 1 from their neighbors
    EXPECT_EQ(m->labels()->tree().getValue(Coord(-1, 5, 5)), 1);
    EXPECT_EQ(m->labels()->tree().getValue(Coord(16, 5, 5)), 2);
    EXPECT_FLOAT_EQ(m->phi()->tree().getValue(Coord(-1, 5, 5)), 3.0f);

    m->erode(2);
    EXPECT_EQ(m->phi()->tree().leafCount(), m->labels()->tree().leafCount());
    EXPECT_TRUE(m->phi()->tree().hasSameTopology(m->labels()->tree()));
    EXPECT_EQ(m->phi()->activeVoxelCount(), n0);
    EXPECT_EQ(m->phi()->activeVoxelCount(), m->labels()->activeVoxelCount());
    EXPECT_TRUE(m->validate());
}

TEST_F(TestLevelSetMultiObject, HolderDilateTieBreak)
{
    for (int swap = 0; swap < 2; ++swap) {
        FloatGrid::Ptr phi = FloatGrid::create(3.0f);
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
        phi->tree().setValueOn(Coord(0, 0, 0), swap ? 0.2f : 0.5f);
        lab->tree().setValueOn(Coord(0, 0, 0), 1);
        phi->tree().setValueOn(Coord(2, 0, 0), swap ? 0.5f : 0.2f);
        lab->tree().setValueOn(Coord(2, 0, 0), 2);
        MLS m(phi, lab, 3);
        m.dilate(1);
        EXPECT_TRUE(m.phi()->tree().isValueOn(Coord(1, 0, 0)));
        EXPECT_EQ(m.labels()->tree().getValue(Coord(1, 0, 0)), swap ? 1 : 2);
        EXPECT_TRUE(m.validate());
    }
}

TEST_F(TestLevelSetMultiObject, HolderErodePrunesInactive)
{
    const double dx = 0.5;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3.0 * dx));
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    const CoordBBox box(Coord(3, 5, 2), Coord(13, 12, 10)); // not leaf aligned
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    auto labelAt = [](const Coord& c) { return c[0] < 8 ? 1 : 2; };
    for (auto it = box.begin(); it; ++it) {
        pa.setValue(*it, 0.25f);
        la.setValue(*it, static_cast<MLS::LabelT>(labelAt(*it)));
    }
    MLS m(phi, lab, 3);
    m.erode(1);
    EXPECT_EQ(m.phi()->tree().leafCount(), m.labels()->tree().leafCount());
    EXPECT_TRUE(m.phi()->tree().hasSameTopology(m.labels()->tree()));
    EXPECT_TRUE(m.validate());
    EXPECT_LT(m.phi()->activeVoxelCount(), Index64(box.volume()));
    const float bg = float(3.0 * dx);
    Index64 off = 0;
    for (auto it = box.begin(); it; ++it) {
        if (m.phi()->tree().isValueOn(*it)) continue;
        ++off;
        EXPECT_FLOAT_EQ(m.phi()->tree().getValue(*it), bg);
        EXPECT_EQ(m.labels()->tree().getValue(*it), labelAt(*it));
    }
    EXPECT_GT(off, Index64(0));
    // every inactive phi voxel in surviving leaves is background
    for (auto leaf = m.phi()->tree().cbeginLeaf(); leaf; ++leaf) {
        for (auto v = leaf->cbeginValueOff(); v; ++v) EXPECT_FLOAT_EQ(*v, bg);
    }
}

namespace {

// Planar interface at x = 7.3 (label 1 for x <= 7, label 2 for x >= 8),
// active band x in [4, 11] (four voxels on each side), y, z in [0, 15].
// Layers x in [7 - halo, 8 + halo] hold the exact distances (halo 0: the two
// interface layers, enough for redistance(); halo 1: also the outer
// neighbors read by the subcell reseed); if @a exact is false the remaining
// band voxels hold garbage (alternating 0.5*dx and halfWidth*dx).
MLS::Ptr
makePlanar(double dx, bool exact, int halo = 1)
{
    const int halfWidth = 3;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = 4; i <= 11; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const Coord c(i, j, k);
                float d = float(std::abs(i - 7.3) * dx);
                if (!exact && (i < 7 - halo || i > 8 + halo)) {
                    d = ((i + j + k) % 2) ? float(0.5 * dx) : float(halfWidth * dx);
                }
                pa.setValue(c, d);
                la.setValue(c, i <= 7 ? 1 : 2);
            }
    // One inactive shell around the band holding the deep-interior labels
    // (inactive labels keep their region's label), with phi leaves kept in lockstep at the background.
    for (int i = 3; i <= 12; ++i)
        for (int j = -1; j <= 16; ++j)
            for (int k = -1; k <= 16; ++k) {
                const Coord c(i, j, k);
                if (pa.isValueOn(c)) continue;
                pa.touchLeaf(c); // inactive phi already holds the background
                la.setValueOff(c, i <= 7 ? 1 : 2);
            }
    return MLS::Ptr(new MLS(phi, lab, halfWidth));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, RedistancePlanar)
{
    const double dx = 0.5;
    for (int reseed = 0; reseed < 2; ++reseed) {
        SCOPED_TRACE(reseed ? "reinitialize" : "redistance");
        auto m = makePlanar(dx, /*exact=*/false, /*halo=*/reseed);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const Index64 nActive = m->phi()->activeVoxelCount();
        const Index64 nLeaves = m->phi()->tree().leafCount();

        tools::MultiLevelSetRedistance<MLS> redist(*m);
        const int iterations = reseed ? redist.reinitialize() : redist.redistance();
        EXPECT_GE(iterations, 1);
        EXPECT_LE(iterations, int(std::ceil(3.0 * std::sqrt(3.0))) + 2);

        EXPECT_EQ(m->phi()->activeVoxelCount(), nActive);
        EXPECT_EQ(m->phi()->tree().leafCount(), nLeaves);
        EXPECT_TRUE(m->validate(&msg)) << msg;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            const Coord c = it.getCoord();
            EXPECT_NEAR(*it, std::abs(c[0] - 7.3) * dx, 1e-3 * dx) << c;
            EXPECT_EQ(m->labels()->tree().getValue(c), c[0] <= 7 ? 1 : 2);
        }
        // inactive phi stays at the background
        for (auto leaf = m->phi()->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto v = leaf->cbeginValueOff(); v; ++v) EXPECT_FLOAT_EQ(*v, float(3.0 * dx));
        }
    }
}

TEST_F(TestLevelSetMultiObject, RedistanceKeepsCrossing)
{
    const double dx = 0.5;
    for (int reseed = 0; reseed < 2; ++reseed) {
        SCOPED_TRACE(reseed ? "reinitialize" : "redistance");
        auto m = makePlanar(dx, /*exact=*/true);
        tools::MultiLevelSetRedistance<MLS> redist(*m);
        if (reseed) redist.reinitialize(); else redist.redistance();
        const auto& t = m->phi()->tree();
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const double a = t.getValue(Coord(7, j, k));
                const double b = t.getValue(Coord(8, j, k));
                ASSERT_GT(a + b, 0.0);
                EXPECT_NEAR(7.0 + a / (a + b), 7.3, 1e-5);
            }
    }
}

namespace {

// Ball of radius R (index units) at @a ctr: label 1 inside, 2 outside; band
// |r - R| <= 3 voxels with phi = |r - R|*dx, and every other voxel of the
// box [lo, hi]^3 inactive with its deep-interior label (inactive labels keep their region's label).
MLS::Ptr
makeBall(double dx, const Vec3d& ctr, double R, int lo, int hi)
{
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3 * dx));
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = lo; i <= hi; ++i)
        for (int j = lo; j <= hi; ++j)
            for (int k = lo; k <= hi; ++k) {
                const Coord c(i, j, k);
                const double r = (Vec3d(i, j, k) - ctr).length();
                const int label = r < R ? 1 : 2;
                if (std::abs(r - R) <= 3.0) {
                    pa.setValue(c, float(std::abs(r - R) * dx));
                    la.setValue(c, static_cast<MLS::LabelT>(label));
                } else {
                    pa.touchLeaf(c);
                    la.setValueOff(c, static_cast<MLS::LabelT>(label));
                }
            }
    return MLS::Ptr(new MLS(phi, lab, 3));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, RedistanceCurvedNoDrift)
{
    const Vec3d ctr(0.3, -0.2, 0.1);
    const double R = 9.7;
    const Coord* offs = tools::lsmo_internal::faceOffsets();

    // redistance() freezes the interface: repeated calls never move it.
    auto m = makeBall(1.0, ctr, R, -15, 15);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const FloatTree initial(m->phi()->tree());
    tools::MultiLevelSetRedistance<MLS> redist(*m);
    for (int n = 0; n < 10; ++n) redist.redistance();
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const auto& labels = m->labels()->tree();
    Index64 nInterface = 0;
    for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6; ++f) iface |= labels.getValue(p + offs[f]) != labels.getValue(p);
        if (!iface) continue;
        ++nInterface;
        const float a = *it, b = initial.getValue(p);
        uint32_t ua, ub;
        std::memcpy(&ua, &a, 4);
        std::memcpy(&ub, &b, 4);
        EXPECT_EQ(ua, ub) << p;
    }
    EXPECT_GT(nInterface, Index64(100));

    // reinitialize() places crossings accurately on a fresh copy.
    auto m2 = makeBall(1.0, ctr, R, -15, 15);
    tools::MultiLevelSetRedistance<MLS>(*m2).reinitialize();
    EXPECT_TRUE(m2->validate(&msg)) << msg;
    const auto& phi2 = m2->phi()->tree();
    const auto& lab2 = m2->labels()->tree();
    double sumErr = 0.0;
    Index64 nPairs = 0;
    for (auto it = phi2.cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        if (lab2.getValue(p) != 1) continue;
        for (int f = 0; f < 6; ++f) {
            const Coord q = p + offs[f];
            if (lab2.getValue(q) != 2 || !phi2.isValueOn(q)) continue;
            const double a = *it, b = phi2.getValue(q);
            const double t = a / (a + b);
            // exact crossing |c + s*e - ctr| = R, s in [0, 1]
            const Vec3d c = Vec3d(p[0], p[1], p[2]) - ctr;
            const Vec3d e(offs[f][0], offs[f][1], offs[f][2]);
            const double ce = c.dot(e);
            const double s = -ce + std::sqrt(ce * ce - (c.lengthSqr() - R * R));
            sumErr += std::abs(t - s);
            ++nPairs;
        }
    }
    ASSERT_GT(nPairs, Index64(100));
    // Regression guard only: the subcell reseed's curvature bias measures
    // 0.0262*dx here, and the exact field scores 0.0109*dx on this metric.
    // The accuracy gates are tested by the round-trip tests.
    EXPECT_LT(sumErr / double(nPairs), 0.03);
}

TEST_F(TestLevelSetMultiObject, RedistanceOrderIndependent)
{
    // Three labels over several leaves: a ball split into labels 1 | 2 by the
    // plane x = 11.5, label 3 outside; phi starts as a rough guess. Inactive
    // voxels of [-1, 24]^3 hold their deep-interior label (inactive labels keep their region's label).
    auto build = []() {
        const double dx = 0.25;
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(float(3 * dx));
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        const Vec3d ctr(11.5, 12.2, 11.7);
        const double R = 7.4;
        for (int i = -1; i <= 24; ++i)
            for (int j = -1; j <= 24; ++j)
                for (int k = -1; k <= 24; ++k) {
                    const double r = (Vec3d(i, j, k) - ctr).length();
                    const Coord c(i, j, k);
                    const int label = r < R ? (i < 12 ? 1 : 2) : 3;
                    const bool inside = i >= 0 && i < 24 && j >= 0 && j < 24 && k >= 0 && k < 24;
                    if (inside && std::abs(r - R) <= 4.0) {
                        pa.setValue(c, float(std::abs(r - R) * dx * (1.0 + 0.1 * ((i * 7 + j * 3 + k) % 5))));
                        la.setValue(c, static_cast<MLS::LabelT>(label));
                    } else {
                        pa.touchLeaf(c);
                        la.setValueOff(c, static_cast<MLS::LabelT>(label));
                    }
                }
        return MLS::Ptr(new MLS(phi, lab, 3));
    };
    for (int reseed = 0; reseed < 2; ++reseed) {
        SCOPED_TRACE(reseed ? "reinitialize" : "redistance");
        auto serial = build();
        auto parallel = build();
        ASSERT_TRUE(serial->validate());
        ASSERT_GT(serial->phi()->tree().leafCount(), Index32(8));

        tools::MultiLevelSetRedistance<MLS> r0(*serial);
        r0.setGrainSize(0);
        const int n0 = reseed ? r0.reinitialize() : r0.redistance();
        tools::MultiLevelSetRedistance<MLS> r1(*parallel);
        r1.setGrainSize(1);
        const int n1 = reseed ? r1.reinitialize() : r1.redistance();
        EXPECT_EQ(n0, n1);
        EXPECT_TRUE(serial->validate());

        ASSERT_TRUE(serial->phi()->tree().hasSameTopology(parallel->phi()->tree()));
        tree::ValueAccessor<const FloatTree> acc(parallel->phi()->tree());
        Index64 n = 0;
        for (auto it = serial->phi()->tree().cbeginValueOn(); it; ++it, ++n) {
            const float a = *it, b = acc.getValue(it.getCoord());
            uint32_t ua, ub;
            std::memcpy(&ua, &a, 4);
            std::memcpy(&ub, &b, 4);
            EXPECT_EQ(ua, ub) << it.getCoord();
        }
        EXPECT_GT(n, Index64(0));
    }
}

TEST_F(TestLevelSetMultiObject, RedistanceSeedInterface)
{
    const double dx = 0.5;
    auto m = makePlanar(dx, /*exact=*/false);
    const FloatTree snapshot(m->phi()->tree());
    FloatTree out(m->phi()->tree());
    BoolTree seeds(false);
    tools::MultiLevelSetRedistance<MLS>::seedInterface(snapshot,
        [](const Coord& c) { return c[0] <= 7; }, out, seeds, dx, /*grainSize=*/0);
    EXPECT_EQ(seeds.activeVoxelCount(), Index64(2 * 16 * 16));
    for (auto it = seeds.cbeginValueOn(); it; ++it) {
        const Coord c = it.getCoord();
        ASSERT_TRUE(c[0] == 7 || c[0] == 8) << c;
        EXPECT_NEAR(out.getValue(c), std::abs(c[0] - 7.3) * dx, 1e-6);
    }
    // non-seed voxels are left alone
    EXPECT_EQ(out.getValue(Coord(5, 3, 3)), snapshot.getValue(Coord(5, 3, 3)));

    // A seed whose only other-side neighbor is inactive has no usable
    // gradient term (Delta would collapse to epsilon): it keeps its value.
    FloatTree lone(float(3 * dx));
    lone.setValueOn(Coord(6, 0, 0), 0.4f);
    lone.setValueOn(Coord(7, 0, 0), 0.15f);
    lone.setValueOff(Coord(8, 0, 0), 0.35f);
    const FloatTree loneSnapshot(lone);
    BoolTree loneSeeds(false);
    tools::MultiLevelSetRedistance<MLS>::seedInterface(loneSnapshot,
        [](const Coord& c) { return c[0] <= 7; }, lone, loneSeeds, dx, /*grainSize=*/0);
    EXPECT_EQ(loneSeeds.activeVoxelCount(), Index64(1));
    EXPECT_TRUE(loneSeeds.isValueOn(Coord(7, 0, 0)));
    EXPECT_EQ(lone.getValue(Coord(7, 0, 0)), 0.15f);
    EXPECT_EQ(lone.getValue(Coord(6, 0, 0)), 0.4f);
}

namespace {

// Interface layer: active voxels with an active face neighbor of opposite
// sign. Errors are |sdf - analytic| in units of dx; analytic takes world
// coordinates.
double
maxInterfaceError(const FloatGrid& sdf, std::function<double(Vec3d)> analytic)
{
    const double dx = sdf.voxelSize()[0];
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    double maxErr = 0.0;
    Index64 n = 0;
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) {
            float v;
            if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
        }
        if (!iface) continue;
        ++n;
        maxErr = std::max(maxErr, std::abs(*it - analytic(sdf.indexToWorld(p))) / dx);
    }
    EXPECT_GT(n, Index64(0));
    return maxErr;
}

// RMS of |sdf - analytic| / dx over all active voxels.
double
rmsBandError(const FloatGrid& sdf, std::function<double(Vec3d)> analytic)
{
    const double dx = sdf.voxelSize()[0];
    double sum = 0.0;
    Index64 n = 0;
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it, ++n) {
        const double e = (*it - analytic(sdf.indexToWorld(it.getCoord()))) / dx;
        sum += e * e;
    }
    EXPECT_GT(n, Index64(0));
    return n ? std::sqrt(sum / double(n)) : 0.0;
}

// Raw signedDistance(ijk, label) over the active voxels of @a m whose world
// position satisfies @a keep.
FloatGrid::Ptr
rawSignedGrid(const MLS& m, int32_t label, std::function<bool(Vec3d)> keep)
{
    FloatGrid::Ptr g = FloatGrid::create(float(m.halfWidth() * m.voxelSize()));
    g->setTransform(m.phi()->transform().copy());
    auto acc = g->getAccessor();
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        if (keep(m.phi()->indexToWorld(p))) acc.setValue(p, m.signedDistance(p, label));
    }
    return g;
}

// Every seed (an active voxel with a face neighbor, active or
// not, of a different label, as the redistancer sees it) lies within one
// voxel of the true interface; distToInterface takes world coordinates.
void
expectNoSpuriousSeeds(const MLS& m, std::function<double(Vec3d)> distToInterface)
{
    const double dx = m.voxelSize();
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    tree::ValueAccessor<const LabelTree> lab(m.labels()->tree());
    Index64 nSeeds = 0, nBad = 0;
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool seed = false;
        for (int f = 0; f < 6; ++f) seed |= lab.getValue(p + offs[f]) != lab.getValue(p);
        if (!seed) continue;
        ++nSeeds;
        const double d = distToInterface(m.phi()->indexToWorld(p));
        if (d > dx * (1.0 + 1e-6)) {
            if (++nBad <= 5) ADD_FAILURE() << "spurious seed at " << p << ", interface " << d / dx << " dx away";
        }
    }
    EXPECT_GT(nSeeds, Index64(0));
    EXPECT_EQ(nBad, Index64(0));
}

struct TwoSphereErrors { double maxIface[2], rms[2]; };

// Analytic signed distance of two-sphere label @a label (1 or 2), world units.
std::function<double(Vec3d)>
twoSphereDistance(double dx, int label)
{
    const double r = 10.0 * dx, c = (label == 2 ? 15.0 : -15.0) * dx;
    return [r, c](Vec3d w) { return (w - Vec3d(c, 0, 0)).length() - r; };
}

// Raw signedDistance errors near each sphere of makeTwoSpheres, in dx.
TwoSphereErrors
twoSphereRoundTrip(double dx)
{
    const double r = 10.0 * dx, cx = 15.0 * dx;
    auto m = makeTwoSpheres(dx);
    EXPECT_TRUE(m);
    if (!m) return TwoSphereErrors{};
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(m->halfWidth(), 3);
    EXPECT_DOUBLE_EQ(m->voxelSize(), dx);
    const auto sphere = [r](double c) {
        return [r, c](Vec3d w) { return (w - Vec3d(c, 0, 0)).length() - r; };
    };
    expectNoSpuriousSeeds(*m, [&](Vec3d w) {
        return std::min(std::abs(sphere(-cx)(w)), std::abs(sphere(cx)(w)));
    });
    TwoSphereErrors e;
    for (int s = 0; s < 2; ++s) {
        const double c = s ? cx : -cx;
        auto g = rawSignedGrid(*m, s + 1, [s](Vec3d w) { return s ? w[0] > 0.0 : w[0] < 0.0; });
        e.maxIface[s] = maxInterfaceError(*g, sphere(c));
        e.rms[s] = rmsBandError(*g, sphere(c));
    }
    return e;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, RoundTripTwoSpheres)
{
    const TwoSphereErrors e = twoSphereRoundTrip(1.0);
    for (int s = 0; s < 2; ++s) {
        std::cout << "RoundTripTwoSpheres label " << s + 1 << ": max interface error "
            << e.maxIface[s] << " dx, RMS band error " << e.rms[s] << " dx" << std::endl;
        EXPECT_LT(e.maxIface[s], 0.05);
        EXPECT_LT(e.rms[s], 0.1);
    }
}

TEST_F(TestLevelSetMultiObject, RoundTripNonUnitVoxel)
{
    const TwoSphereErrors e1 = twoSphereRoundTrip(1.0);
    const TwoSphereErrors e = twoSphereRoundTrip(0.25);
    for (int s = 0; s < 2; ++s) {
        std::cout << "RoundTripNonUnitVoxel label " << s + 1 << ": max interface error "
            << e.maxIface[s] << " dx, RMS band error " << e.rms[s] << " dx" << std::endl;
        EXPECT_LT(e.maxIface[s], 0.05);
        EXPECT_LT(e.rms[s], 0.1);
        EXPECT_NEAR(e.maxIface[s], e1.maxIface[s], 1e-3);
        EXPECT_NEAR(e.rms[s], e1.rms[s], 1e-3);
    }
}

TEST_F(TestLevelSetMultiObject, ConvergenceOrder)
{
    const double r = 0.5;
    const double dxs[3] = {1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0};
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (double dx : dxs) {
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(0, 0, 0), float(dx));
        auto m = MLS::fromSignedGrids(in);
        ASSERT_TRUE(m);
        ASSERT_TRUE(m->validate());
        auto g = rawSignedGrid(*m, 1, [](Vec3d) { return true; });
        const double errWorld = dx * maxInterfaceError(*g, [r](Vec3d w) { return w.length() - r; });
        std::cout << "ConvergenceOrder dx = 1/" << int(std::lround(1.0 / dx))
            << ": max interface error " << errWorld << " (" << errWorld / dx << " dx)" << std::endl;
        ASSERT_GT(errWorld, 0.0);
        const double x = std::log(dx), y = std::log(errWorld);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double n = 3.0;
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    std::cout << "ConvergenceOrder slope " << slope << std::endl;
    EXPECT_GE(slope, 1.8);
}

TEST_F(TestLevelSetMultiObject, FromLabelGrid)
{
    // Label 1 on [5, 24]^3 inside label 2 on [0, 29]^3; label 0 elsewhere.
    const double dx = 0.5;
    Int32Grid::Ptr in = Int32Grid::create(0);
    in->setTransform(math::Transform::createLinearTransform(dx));
    in->tree().fill(CoordBBox(Coord(0), Coord(29)), 2, /*active=*/true);
    in->tree().fill(CoordBBox(Coord(5), Coord(24)), 1, /*active=*/true);
    in->tree().voxelizeActiveTiles();

    auto m = MLS::fromLabelGrid(*in);
    ASSERT_TRUE(m);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(m->halfWidth(), 3);
    EXPECT_DOUBLE_EQ(m->voxelSize(), dx);

    // Index-space distance from a voxel center to the face planes of a box
    // whose voxels span [lo, hi] (faces at lo - 0.5 and hi + 0.5).
    const auto boxDist = [](const Vec3d& p, double lo, double hi) {
        const double l = lo - 0.5, h = hi + 0.5;
        double inner = std::numeric_limits<double>::max();
        bool inside = true;
        Vec3d q(0.0);
        for (int a = 0; a < 3; ++a) {
            q[a] = std::max(0.0, std::max(l - p[a], p[a] - h));
            if (p[a] < l || p[a] > h) inside = false;
            inner = std::min(inner, std::min(p[a] - l, h - p[a]));
        }
        return inside ? inner : q.length();
    };
    expectNoSpuriousSeeds(*m, [&](Vec3d w) {
        const Vec3d p = w / dx;
        return dx * std::min(boxDist(p, 5, 24), boxDist(p, 0, 29));
    });

    // Every interface voxel keeps its frozen face-midpoint seed, 0.5*dx
    // (fromLabelGrid redistances without reseeding).
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    const auto& lab = m->labels()->tree();
    Index64 nFace = 0;
    for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        EXPECT_EQ(lab.getValue(p), in->tree().getValue(p)) << p;
        bool iface = false;
        for (int f = 0; f < 6; ++f) iface |= lab.getValue(p + offs[f]) != lab.getValue(p);
        if (!iface) continue;
        ++nFace;
        EXPECT_NEAR(*it, 0.5 * dx, 1e-3 * dx) << p;
    }
    EXPECT_GT(nFace, Index64(6 * 18 * 18));
    // band: every active voxel within halfWidth*dx (plus solver tolerance)
    for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) EXPECT_LE(*it, float(3.0 * dx));
}

TEST_F(TestLevelSetMultiObject, FromSignedGridsThrows)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    FloatGrid::Ptr fog = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(0), 1.0f);
    fog->setGridClass(GRID_FOG_VOLUME);
    in[1] = fog;
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);

    FloatGrid::Ptr a = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(-8, 0, 0), 1.0f);
    FloatGrid::Ptr b = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(8, 0, 0), 1.0f);
    FloatGrid::Ptr c = tools::createLevelSetSphere<FloatGrid>(2.5f, Vec3f(4, 0, 0), 0.5f);

    in.clear(); // empty map
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in[1] = a; in[2] = FloatGrid::ConstPtr(); // null grid
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[0] = a; in[2] = b; // key 0 is the background
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[1] = a; in[2] = c; // mismatched transforms
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[1] = a; in[2] = b;
    EXPECT_THROW(MLS::fromSignedGrids(in, 1), ValueError); // halfWidth < 2
    EXPECT_NO_THROW(MLS::fromSignedGrids(in, 2));
}

namespace {

void
expectLevelSetGrid(const FloatGrid& g, const MLS& m, int halfWidth)
{
    EXPECT_EQ(g.getGridClass(), GRID_LEVEL_SET);
    EXPECT_FLOAT_EQ(g.background(), float(halfWidth * m.voxelSize()));
    EXPECT_EQ(g.transform(), m.phi()->transform());
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, TripleJunction)
{
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    const std::vector<TripleJunction> set = junctionSet();
    double worstFar = 0.0, worstNear = 0.0, worstRms = 0.0;
    for (size_t n = 0; n < set.size(); ++n) {
        const TripleJunction& tj = set[n];
        const double dx = tj.dx;
        auto m = tj.build();
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));

        for (int l = 1; l <= 3; ++l) {
            SCOPED_TRACE("junction " + std::to_string(n) + " label " + std::to_string(l));
            const auto analytic = [&tj, l](Vec3d w) { return tj.signedDist(w, l); };

            FloatGrid::Ptr g = m->extractSignedDistance(l);
            ASSERT_TRUE(g);
            expectLevelSetGrid(*g, *m, 3);
            const double rms = rmsBandError(*g, analytic);

            // Interface-layer error binned by distance to the junction line.
            const double edges[4] = {1.0, 2.0, 4.0, 1e9};
            double binMax[4] = {0, 0, 0, 0};
            Index64 binN[4] = {0, 0, 0, 0};
            tree::ValueAccessor<const FloatTree> acc(g->tree());
            for (auto it = g->tree().cbeginValueOn(); it; ++it) {
                const Coord p = it.getCoord();
                bool iface = false;
                for (int f = 0; f < 6 && !iface; ++f) {
                    float v;
                    if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
                }
                if (!iface) continue;
                const Vec3d w = g->indexToWorld(p);
                int b = 0;
                while (tj.lineDist(w) >= edges[b]) ++b;
                ++binN[b];
                binMax[b] = std::max(binMax[b], std::abs(*it - analytic(w)) / dx);
            }
            // Gates (first order near the line: psi_l is not smooth there):
            // < 0.05*dx at least 2 voxels from the line, < 0.5*dx within 2
            // voxels, RMS over the whole band < 0.1*dx.
            const double farMax = std::max(binMax[2], binMax[3]);
            const double nearMax = std::max(binMax[0], binMax[1]);
            std::cout << "TripleJunction " << n << " label " << l << ": interface max >=2 vox "
                << farMax << " dx, <2 vox " << nearMax << " dx, RMS " << rms
                << " dx; by line distance [<1: " << binMax[0] << " (" << binN[0] << ")"
                << ", 1-2: " << binMax[1] << " (" << binN[1] << ")"
                << ", 2-4: " << binMax[2] << " (" << binN[2] << ")"
                << ", >4: " << binMax[3] << " (" << binN[3] << ")]" << std::endl;
            EXPECT_LT(farMax, 0.05);
            EXPECT_LT(nearMax, 0.5);
            EXPECT_LT(rms, 0.1);
            worstFar = std::max(worstFar, farMax);
            worstNear = std::max(worstNear, nearMax);
            worstRms = std::max(worstRms, rms);

            // Every voxel within halfWidth of the label boundary is in the
            // band (checked 3+ voxels inside the fixture box, beyond which
            // the boundary's nearest point may lie outside the truncated band).
            for (auto it = g->tree().cbeginValueOn(); it; ++it) EXPECT_LT(std::abs(*it), 3.0 * dx);
            Index64 missing = 0;
            for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
                const Coord c = it.getCoord();
                if (tj.boxDepth(c) < 3.0) continue;
                const Vec3d w = m->phi()->indexToWorld(c);
                if (std::abs(analytic(w)) < 2.5 * dx && !g->tree().isValueOn(c)) ++missing;
            }
            EXPECT_EQ(missing, Index64(0));

            // Regression guard: raw +-phi underestimates psi_l
            // outside l near the junction.
            auto raw = rawSignedGrid(*m, l, [&tj, l](Vec3d w) { return tj.sector(w) != l; });
            double maxUnder = 0.0;
            for (auto it = raw->tree().cbeginValueOn(); it; ++it) {
                const Vec3d w = raw->indexToWorld(it.getCoord());
                maxUnder = std::max(maxUnder, (analytic(w) - *it) / dx);
            }
            EXPECT_GT(maxUnder, 0.2);
        }
    }
    std::cout << "TripleJunction over " << set.size() << " junctions: worst interface max >=2 vox "
        << worstFar << " dx, <2 vox " << worstNear << " dx, worst RMS " << worstRms << " dx" << std::endl;
}

namespace {

/// Max interface-layer error of @a g against the analytic distance to
/// sector @a l of @a tj, in voxels, within (@a nearMax) and at least
/// (@a farMax) 2 voxels from the junction line.
template<typename JunctionT>
void
junctionInterfaceError(const FloatGrid& g, const JunctionT& tj, int l,
    double& nearMax, double& farMax)
{
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    constexpr double kNearLineVoxels = 2.0;
    nearMax = farMax = 0.0;
    tree::ValueAccessor<const FloatTree> acc(g.tree());
    for (auto it = g.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) {
            float v;
            if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
        }
        if (!iface) continue;
        const Vec3d w = g.indexToWorld(p);
        const double e = std::abs(*it - tj.signedDist(w, l)) / tj.dx;
        double& bin = tj.lineDist(w) < kNearLineVoxels ? nearMax : farMax;
        bin = std::max(bin, e);
    }
}

/// Largest |a - b| / dx over the interface voxels of @a a (sign change to a
/// face neighbor) at least 2 voxels from the junction line of @a tj.
template<typename JunctionT>
double
farFieldDifference(const FloatGrid& a, const FloatGrid& b, const JunctionT& tj)
{
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    double worst = 0.0;
    tree::ValueAccessor<const FloatTree> acc(a.tree()), bAcc(b.tree());
    for (auto it = a.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) {
            float v;
            if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
        }
        if (!iface || tj.lineDist(a.indexToWorld(p)) < 2.0) continue;
        worst = std::max(worst, std::abs(double(*it) - double(bAcc.getValue(p))) / tj.dx);
    }
    return worst;
}

/// Three labels around a z-parallel line through index point (c, 0) with
/// arbitrary angular spans (degrees, summing to 360): label l owns the
/// polar angles [b_{l-1}, b_l) about the line, b_0 = theta0. Same analytic
/// distance and pair construction as lsmo_test::TripleJunction.
struct SectorJunction
{
    double dx = 0.5;
    double b[3];      // start angle of each sector, radians
    double span01[2]; // cumulative end angles of sectors 1 and 2 past b[0]
    Vec2d c;
    int H = 12;

    SectorJunction(double theta0Deg, const double spanDeg[3], const Vec2d& center) : c(center)
    {
        const double rad = math::pi<double>() / 180.0;
        b[0] = theta0Deg * rad;
        b[1] = b[0] + spanDeg[0] * rad;
        b[2] = b[1] + spanDeg[1] * rad;
        span01[0] = spanDeg[0] * rad;
        span01[1] = (spanDeg[0] + spanDeg[1]) * rad;
    }
    Vec2d local(const Vec3d& w) const { return Vec2d(w[0] / dx - c[0], w[1] / dx - c[1]); }
    Vec2d ray(int k) const { return Vec2d(std::cos(b[k % 3]), std::sin(b[k % 3])); }
    int sector(const Vec3d& w) const
    {
        const Vec2d p = local(w);
        const double twoPi = 2.0 * math::pi<double>();
        double t = std::atan2(p[1], p[0]) - b[0];
        while (t < 0.0) t += twoPi;
        while (t >= twoPi) t -= twoPi;
        return t < span01[0] ? 1 : (t < span01[1] ? 2 : 3);
    }
    double signedDist(const Vec3d& w, int l) const
    {
        const Vec2d p = local(w);
        const double d = std::min(TripleJunction::rayDist(p, ray(l - 1)),
            TripleJunction::rayDist(p, ray(l)));
        return dx * (sector(w) == l ? -d : d);
    }
    double lineDist(const Vec3d& w) const { return local(w).length(); }
    double boxDepth(const Coord& p) const
    {
        return H - std::max({std::abs(p[0] + 0.5), std::abs(p[1] + 0.5), std::abs(p[2] + 0.5)});
    }
    MLS::Ptr build() const
    {
        const int halfWidth = 3;
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        for (int i = -H - 1; i <= H; ++i)
            for (int j = -H - 1; j <= H; ++j)
                for (int k = -H - 1; k <= H; ++k) {
                    const Coord q(i, j, k);
                    const Vec3d w = xform->indexToWorld(q);
                    const int l = sector(w);
                    const double d = std::abs(signedDist(w, l));
                    if (boxDepth(q) > 0.0 && d < halfWidth * dx) {
                        pa.setValue(q, float(d));
                        la.setValue(q, static_cast<MLS::LabelT>(l));
                    } else {
                        pa.touchLeaf(q);
                        la.setValueOff(q, static_cast<MLS::LabelT>(l));
                    }
                }
        return MLS::Ptr(new MLS(phi, lab, halfWidth));
    }
};

/// Outside voxels of @a l that extractSignedDistance rebuilds: active, not
/// @a l, with a face neighbor of @a l and one of a third label.
std::vector<Coord>
junctionSeedsOutside(const MLS& m, int l)
{
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    std::vector<Coord> out;
    auto lab = m.labels()->getConstAccessor();
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord q = it.getCoord();
        const int lq = lab.getValue(q);
        if (lq == l) continue;
        bool nl = false, third = false;
        for (int f = 0; f < 6; ++f) {
            const int ln = lab.getValue(q + offs[f]);
            nl = nl || ln == l;
            third = third || (ln != l && ln != lq);
        }
        if (nl && third) out.push_back(q);
    }
    return out;
}

} // anonymous namespace

// Closest-point reconstruction next to a reflex (240 degree) sector, where
// the region is locally a union of half-spaces and no boundary plane
// supports it: the union distance must decide some seeds. For every label
// (the 60 degree slivers included) the result must be no worse than the
// Godunov rule within 2 voxels of the line and below 0.5*dx there, and the
// far field must not change.
TEST_F(TestLevelSetMultiObject, ClosestPointReflexJunction)
{
    const double spans[3] = {240.0, 60.0, 60.0};
    const SectorJunction set[2] = {SectorJunction(25.0, spans, Vec2d(0.37, -0.21)),
        SectorJunction(200.0, spans, Vec2d(-0.13, 0.29))};
    constexpr double kFarTolerance = 1e-6;
    for (int n = 0; n < 2; ++n) {
        const SectorJunction& sj = set[n];
        auto m = sj.build();
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));
        for (int l = 1; l <= 3; ++l) {
            SCOPED_TRACE("reflex junction " + std::to_string(n) + " label " + std::to_string(l));
            tools::ExtractSettings s;
            s.closestPointJunctions = false;
            FloatGrid::Ptr off = m->extractSignedDistance(l, s);
            s.closestPointJunctions = true;
            FloatGrid::Ptr on = m->extractSignedDistance(l, s);
            double nearOff, farOff, nearOn, farOn;
            junctionInterfaceError(*off, sj, l, nearOff, farOff);
            junctionInterfaceError(*on, sj, l, nearOn, farOn);
            // the union distance decides at least one rebuilt seed of the reflex sector
            int decisive = 0;
            tree::ValueAccessor<const FloatTree, false> pa(m->phi()->tree());
            tree::ValueAccessor<const LabelTree, false> la(m->labels()->tree());
            const std::vector<Coord> seeds = junctionSeedsOutside(*m, l);
            for (const Coord& q : seeds) {
                tools::lsmo_internal::ClosestPointInfo info;
                tools::lsmo_internal::closestPointPsi(pa, la, q, l, sj.dx, 0.0, &info);
                decisive += info.unionDecisive ? 1 : 0;
            }
            std::cout << "[ClosestPointReflexJunction] junction " << n << " label " << l
                      << " (span " << spans[l - 1] << " deg): <2 vox off " << nearOff
                      << " dx, on " << nearOn << " dx; >=2 vox off " << farOff << " dx, on "
                      << farOn << " dx; union-decided seeds " << decisive << " of "
                      << seeds.size() << std::endl;
            constexpr double kNearGate = 0.5;
            EXPECT_LE(farOn, 0.05);
            EXPECT_LE(farFieldDifference(*on, *off, sj), kFarTolerance);
            EXPECT_LE(nearOn, nearOff + kFarTolerance);
            EXPECT_LT(nearOn, kNearGate);
            if (l == 1) { EXPECT_GT(decisive, 0); } // the 240 degree sector
        }
    }
}

// The phi floor of the accessor form of closestPointPsi: a floor below
// every phi changes nothing; a floor above some phi gives the same result
// as flooring the tree itself, and never less than the floor.
TEST_F(TestLevelSetMultiObject, ClosestPointPsiFloor)
{
    const double spans[3] = {240.0, 60.0, 60.0};
    const TripleJunction tj = junctionSet()[8];
    const SectorJunction sj(25.0, spans, Vec2d(0.37, -0.21));
    MLS::Ptr ms[2] = {tj.build(), sj.build()};
    const double dx = tj.dx;
    for (int n = 0; n < 2; ++n) {
        const MLS& m = *ms[n];
        const FloatTree& phi = m.phi()->tree();
        float minPhi = std::numeric_limits<float>::max();
        for (auto it = phi.cbeginValueOn(); it; ++it) minPhi = std::min(minPhi, *it);
        ASSERT_GT(minPhi, 0.0f);
        const double lowFloor = 0.5 * double(minPhi);
        const double highFloor = 0.25 * dx;
        FloatTree floored(phi);
        for (auto it = floored.beginValueOn(); it; ++it) {
            it.setValue(std::max(*it, float(highFloor)));
        }
        tree::ValueAccessor<const FloatTree, false> pa(phi), fa(floored);
        tree::ValueAccessor<const LabelTree, false> la(m.labels()->tree());
        int raised = 0, total = 0;
        for (int l = 1; l <= 3; ++l) {
            for (const Coord& q : junctionSeedsOutside(m, l)) {
                const float ref = tools::lsmo_internal::closestPointPsi(pa, la, q, l, dx, 0.0);
                EXPECT_EQ(ref, tools::lsmo_internal::closestPointPsi(
                    phi, m.labels()->tree(), q, l, dx));
                EXPECT_EQ(ref, tools::lsmo_internal::closestPointPsi(pa, la, q, l, dx, lowFloor));
                const float high = tools::lsmo_internal::closestPointPsi(pa, la, q, l, dx,
                    highFloor);
                EXPECT_EQ(high, tools::lsmo_internal::closestPointPsi(fa, la, q, l, dx, 0.0));
                EXPECT_GE(double(high), highFloor * (1.0 - 1e-6));
                raised += high != ref ? 1 : 0;
                ++total;
            }
        }
        std::cout << "[ClosestPointPsiFloor] fixture " << n << ": min phi " << minPhi / dx
                  << " dx; floor " << highFloor / dx << " dx changed " << raised << " of "
                  << total << " estimates" << std::endl;
        EXPECT_GT(total, 0);
        EXPECT_GT(raised, 0);
    }
}

// Closest-point junction reconstruction (ExtractSettings::closestPointJunctions):
// on (the default) lowers the worst interface error within 2 voxels of the
// junction line over the set and leaves the far field exact; off reproduces
// the Godunov junction rule (worst-case numbers per junction recorded before
// the option existed).
TEST_F(TestLevelSetMultiObject, ClosestPointJunctions)
{
    EXPECT_TRUE(tools::ExtractSettings().closestPointJunctions);
    EXPECT_EQ(tools::ExtractSettings().halfWidth, -1);
    EXPECT_EQ(tools::ExtractSettings().grainSize, 1);

    // worst <2-voxel interface error over labels 1..3 per junction, option off
    const double kOffNear[12] = {0.187471, 0.330579, 0.214058, 0.182584, 0.191706, 0.0826463,
        0.173923, 0.122111, 0.376864, 0.287112, 0.362498, 0.340601};
    constexpr double kRecordedTolerance = 1e-5; // printed precision of the record
    constexpr double kFarGate = 0.05;
    const std::vector<TripleJunction> set = junctionSet();
    ASSERT_EQ(set.size(), size_t(12));
    double worst[2] = {0.0, 0.0}, worstFar[2] = {0.0, 0.0};
    for (size_t n = 0; n < set.size(); ++n) {
        const TripleJunction& tj = set[n];
        auto m = tj.build();
        double near[2] = {0.0, 0.0};
        for (int on = 0; on < 2; ++on) {
            tools::ExtractSettings s;
            s.closestPointJunctions = on == 1;
            for (int l = 1; l <= 3; ++l) {
                FloatGrid::Ptr g = m->extractSignedDistance(l, s);
                ASSERT_TRUE(g);
                expectLevelSetGrid(*g, *m, 3);
                double nm, fm;
                junctionInterfaceError(*g, tj, l, nm, fm);
                near[on] = std::max(near[on], nm);
                worstFar[on] = std::max(worstFar[on], fm);
                if (on == 1) {
                    // far field: identical to the option off
                    tools::ExtractSettings so = s;
                    so.closestPointJunctions = false;
                    FloatGrid::Ptr off = m->extractSignedDistance(l, so);
                    EXPECT_LE(farFieldDifference(*g, *off, tj), 1e-6);
                    // the default overload is the option on
                    FloatGrid::Ptr d = m->extractSignedDistance(l);
                    double dn, df;
                    junctionInterfaceError(*d, tj, l, dn, df);
                    EXPECT_EQ(dn, nm);
                    EXPECT_EQ(df, fm);
                }
            }
            worst[on] = std::max(worst[on], near[on]);
        }
        std::cout << "[ClosestPointJunctions] junction " << n << ": <2 vox off " << near[0]
                  << " dx, on " << near[1] << " dx" << std::endl;
        EXPECT_NEAR(near[0], kOffNear[n], kRecordedTolerance) << "junction " << n;
    }
    std::cout << "[ClosestPointJunctions] worst <2 vox off " << worst[0] << " dx, on "
              << worst[1] << " dx; worst >=2 vox off " << worstFar[0] << " dx, on "
              << worstFar[1] << " dx" << std::endl;
    EXPECT_NEAR(worst[0], 0.376864, kRecordedTolerance);
    EXPECT_LT(worst[1], worst[0]);
    EXPECT_LE(worstFar[1], kFarGate);
    EXPECT_LE(worstFar[0], kFarGate);
}

TEST_F(TestLevelSetMultiObject, ExtractMatchesRoundTrip)
{
    const double dx = 1.0;
    auto m = makeTwoSpheres(dx);
    ASSERT_TRUE(m);
    for (int l = 1; l <= 2; ++l) {
        SCOPED_TRACE("label " + std::to_string(l));
        const auto analytic = twoSphereDistance(dx, l);
        auto raw = rawSignedGrid(*m, l, [l](Vec3d w) { return l == 2 ? w[0] > 0.0 : w[0] < 0.0; });
        const double rawMax = maxInterfaceError(*raw, analytic);
        const double rawRms = rmsBandError(*raw, analytic);

        FloatGrid::Ptr g = m->extractSignedDistance(l);
        ASSERT_TRUE(g);
        expectLevelSetGrid(*g, *m, 3);
        const double extMax = maxInterfaceError(*g, analytic);
        const double extRms = rmsBandError(*g, analytic);
        std::cout << "ExtractMatchesRoundTrip label " << l << ": max interface error raw "
            << rawMax << " dx, extracted " << extMax << " dx; RMS raw " << rawRms
            << " dx, extracted " << extRms << " dx" << std::endl;
        EXPECT_LE(extMax, rawMax + 1e-4);
        EXPECT_LE(extRms, rawRms + 1e-4);
        // the sign is right everywhere and deep values have the right sign
        for (auto it = g->tree().cbeginValueOn(); it; ++it) {
            const Vec3d w = g->indexToWorld(it.getCoord());
            if (std::abs(analytic(w)) > 0.5 * dx) { EXPECT_EQ(*it < 0.0f, analytic(w) < 0.0); }
        }
        EXPECT_LT(g->tree().getValue(g->transform().worldToIndexCellCentered(
            Vec3d(l == 2 ? 15.0 : -15.0, 0, 0))), 0.0f);
        EXPECT_GT(g->tree().getValue(Coord(0, 0, 0)), 0.0f);

        // A wider band than the pair's is dilated and solved.
        FloatGrid::Ptr wide = m->extractSignedDistance(l, 5);
        ASSERT_TRUE(wide);
        expectLevelSetGrid(*wide, *m, 5);
        EXPECT_GT(wide->activeVoxelCount(), g->activeVoxelCount());
        for (auto it = wide->tree().cbeginValueOn(); it; ++it) EXPECT_LT(std::abs(*it), 5.0 * dx);
        const double wideRms = rmsBandError(*wide, analytic);
        std::cout << "ExtractMatchesRoundTrip label " << l << ": halfWidth 5 RMS "
            << wideRms << " dx" << std::endl;
        EXPECT_LT(wideRms, 0.1);
    }
    EXPECT_THROW(m->extractSignedDistance(1, 1), ValueError);
}

TEST_F(TestLevelSetMultiObject, ExtractAbsentLabel)
{
    auto m = makeTwoSpheres(0.5);
    ASSERT_TRUE(m);
    FloatGrid::Ptr g = m->extractSignedDistance(99);
    ASSERT_TRUE(g);
    expectLevelSetGrid(*g, *m, 3);
    EXPECT_EQ(g->activeVoxelCount(), Index64(0));
    EXPECT_GT(g->background(), 0.0f);
    EXPECT_GT(g->tree().getValue(Coord(-30, 0, 0)), 0.0f);
}

// ---------------------------------------------------------------------------
// LabelAwareStencil and MultiObjectForces
// ---------------------------------------------------------------------------

namespace {

using Stencil = tools::LabelAwareStencil<MLS>;
using Forces = tools::MultiObjectForces<MLS>;

// First active voxel of label 1 whose +x neighbor is label 2 (interface).
Coord
sphereInterfaceVoxel(const MLS& m)
{
    // Sphere 1 is centered at (-15,0,0) dx with radius 10 dx: its surface
    // point towards -y is at (-15, -10, 0).
    const Coord c(-15, -10, 0);
    for (int dj = -1; dj <= 1; ++dj) {
        const Coord p = c.offsetBy(0, dj, 0);
        if (m.labels()->tree().getValue(p) == 1
            && m.labels()->tree().getValue(p.offsetBy(0, -1, 0)) != 1) return p;
    }
    return c;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, StencilSphere)
{
    const double dx = 1.0, r = 10.0;
    auto m = makeTwoSpheres(dx);
    const Coord ijk = sphereInterfaceVoxel(*m);
    ASSERT_EQ(m->labels()->tree().getValue(ijk), 1);
    Stencil s(*m);
    s.moveTo(ijk, 1);
    const Vec3f g = s.gradient();
    EXPECT_NEAR(g.length(), 1.0f, 0.05f);
    // Outward normal at the -y pole of sphere 1 is -y.
    EXPECT_LT(g[1], -0.9f);
    // Single-voxel curvature carries second-difference noise from the
    // redistanced phi, so check the mean over the interface layer of the
    // sphere (voxels of label 1 with a face neighbor of another label).
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    double sum = 0.0;
    int n = 0;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it != 1) continue;
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) iface = m->labels()->tree().getValue(p + offs[f]) != 1;
        if (!iface || p[0] > -10) continue; // sphere-1 side away from the junction with sphere 2
        s.moveTo(p, 1);
        sum += s.meanCurvatureNormGrad();
        ++n;
    }
    ASSERT_GT(n, 100);
    EXPECT_NEAR(sum / n, 1.0 / r, 0.1 / r);
}

TEST_F(TestLevelSetMultiObject, StencilLabelSign)
{
    auto m = makeTwoSpheres(1.0);
    const Coord ijk(15, 0, 0); // center of sphere 2
    ASSERT_EQ(m->labels()->tree().getValue(ijk), 2);
    const float phi = m->phi()->tree().getValue(ijk);
    ASSERT_GT(phi, 0.0f);
    Stencil s(*m);
    s.moveTo(ijk, 1);
    EXPECT_EQ((s.getValue<0, 0, 0>()), phi);
    s.moveTo(ijk, 2);
    EXPECT_EQ((s.getValue<0, 0, 0>()), -phi);
}

TEST_F(TestLevelSetMultiObject, ForcesPressureRescale)
{
    FloatGrid::Ptr p = FloatGrid::create(0.0f);
    auto a = p->getAccessor();
    a.setValue(Coord(0, 0, 0), -2.0f);
    a.setValue(Coord(1, 0, 0), 0.0f);
    a.setValue(Coord(2, 0, 0), 4.0f);
    Forces f;
    f.setPressure(p, 1.0f, 0.0f);
    ASSERT_TRUE(f.pressure());
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(0, 0, 0)), -1.0f);
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(1, 0, 0)), 0.0f);
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(2, 0, 0)), 1.0f);
    // Input is untouched.
    EXPECT_FLOAT_EQ(p->tree().getValue(Coord(2, 0, 0)), 4.0f);
    // NaN target: no rescale.
    Forces g;
    g.setPressure(p, 1.0f);
    EXPECT_FLOAT_EQ(g.pressure()->tree().getValue(Coord(2, 0, 0)), 4.0f);
}

TEST_F(TestLevelSetMultiObject, ForcesMatchesWedgeFormula)
{
    // 3^3 neighborhood, all label 1, so psi = -phi everywhere.
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    float v[3][3][3]; // v[i+1][j+1][k+1] = psi
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k) {
                const float fi = float(i), fj = float(j), fk = float(k);
                const float d = 0.8f + 0.35f * fi - 0.2f * fj + 0.1f * fk + 0.07f * fi * fi
                    + 0.05f * fj * fj - 0.03f * fk * fk + 0.04f * fi * fj - 0.02f * fi * fk
                    + 0.06f * fj * fk;
                pa.setValue(Coord(i, j, k), d);
                la.setValue(Coord(i, j, k), 1);
                v[i + 1][j + 1][k + 1] = -d;
            }
    MLS mls(phi, lab, 3);

    const Vec3f vel(0.3f, -0.4f, 0.2f);
    Vec3SGrid::Ptr vg = Vec3SGrid::create(Vec3f(0.0f));
    vg->tree().setValue(Coord(0, 0, 0), vel);
    FloatGrid::Ptr pg = FloatGrid::create(0.0f);
    pg->tree().setValue(Coord(0, 0, 0), 0.7f);

    const float wc = 0.5f, wa = 1.5f, wp = 2.0f;
    Forces f;
    f.setSpeedExtension(false); // the formula samples the fields at the voxel center
    f.setCurvatureWeight(wc);
    f.setAdvection(vg, wa);
    f.setPressure(pg, wp); // NaN target: values used as given

    Stencil s(mls);
    s.moveTo(Coord(0, 0, 0), 1);
    const float got = f(Coord(0, 0, 0), 1, s);

    // Reference pressure and advection motion, evaluated by hand (dx = 1).
    auto V = [&](int i, int j, int k) { return v[i + 1][j + 1][k + 1]; };
    const float v111 = V(0, 0, 0);
    const float DxNeg = v111 - V(-1, 0, 0), DxPos = V(1, 0, 0) - v111;
    const float DyNeg = v111 - V(0, -1, 0), DyPos = V(0, 1, 0) - v111;
    const float DzNeg = v111 - V(0, 0, -1), DzPos = V(0, 0, 1) - v111;
    auto mn = [](float a) { return std::min(a, 0.0f); };
    auto mx = [](float a) { return std::max(a, 0.0f); };
    const float gsPos = mx(DxNeg) * mx(DxNeg) + mn(DxPos) * mn(DxPos) + mx(DyNeg) * mx(DyNeg)
        + mn(DyPos) * mn(DyPos) + mx(DzNeg) * mx(DzNeg) + mn(DzPos) * mn(DzPos);
    const float gsNeg = mx(DxPos) * mx(DxPos) + mn(DxNeg) * mn(DxNeg) + mx(DyPos) * mx(DyPos)
        + mn(DyNeg) * mn(DyNeg) + mx(DzPos) * mx(DzPos) + mn(DzNeg) * mn(DzNeg);
    const float Dx = 0.5f * (V(1, 0, 0) - V(-1, 0, 0));
    const float Dy = 0.5f * (V(0, 1, 0) - V(0, -1, 0));
    const float Dz = 0.5f * (V(0, 0, 1) - V(0, 0, -1));
    const float Dxx = V(1, 0, 0) - 2 * v111 + V(-1, 0, 0);
    const float Dyy = V(0, 1, 0) - 2 * v111 + V(0, -1, 0);
    const float Dzz = V(0, 0, 1) - 2 * v111 + V(0, 0, -1);
    const float Dxy = (V(1, 1, 0) - V(-1, 1, 0) - V(1, -1, 0) + V(-1, -1, 0)) * 0.25f;
    const float Dxz = (V(1, 0, 1) - V(-1, 0, 1) - V(1, 0, -1) + V(-1, 0, -1)) * 0.25f;
    const float Dyz = (V(0, 1, 1) - V(0, -1, 1) - V(0, 1, -1) + V(0, -1, -1)) * 0.25f;
    const float numer = 0.5f * ((Dyy + Dzz) * Dx * Dx + (Dxx + Dzz) * Dy * Dy
        + (Dxx + Dyy) * Dz * Dz - 2 * Dx * Dy * Dxy - 2 * Dx * Dz * Dxz - 2 * Dy * Dz * Dyz);
    const float denom = Dx * Dx + Dy * Dy + Dz * Dz;
    // interface curvature k_x/(1 - psi*k_x) (the reference uses k_x)
    float kn = numer / denom;
    const float pk = v111 * kn / std::sqrt(denom);
    EXPECT_LT(std::abs(pk), 0.5f); // the correction branch is exercised
    kn /= (1.0f - pk);
    float kappa = wc * kn;
    kappa = std::max(-10.0f, std::min(10.0f, kappa));
    const float fx = wa * vel[0], fy = wa * vel[1], fz = wa * vel[2];
    float adv = 0;
    adv += fx > 0 ? fx * DxNeg : (fx < 0 ? fx * DxPos : 0.0f);
    adv += fy > 0 ? fy * DyNeg : (fy < 0 ? fy * DyPos : 0.0f);
    adv += fz > 0 ? fz * DzNeg : (fz < 0 ? fz * DzPos : 0.0f);
    const float F = wp * 0.7f;
    float press = 0;
    if (F > 0) press = -F * std::sqrt(gsPos);
    else if (F < 0) press = -F * std::sqrt(gsNeg);
    const float expected = -adv + kappa + press;
    EXPECT_NEAR(got, expected, 1e-6f);
    // Make sure every term is exercised.
    EXPECT_GT(std::abs(adv), 1e-3f);
    EXPECT_GT(std::abs(kappa), 1e-3f);
    EXPECT_GT(std::abs(press), 1e-3f);

    // Negative pressure force takes the other Godunov branch.
    pg->tree().setValue(Coord(0, 0, 0), -0.7f);
    Forces f2;
    f2.setSpeedExtension(false);
    f2.setPressure(pg, wp);
    const float got2 = f2(Coord(0, 0, 0), 1, s);
    EXPECT_NEAR(got2, 1.4f * std::sqrt(gsNeg), 1e-6f);
}

TEST_F(TestLevelSetMultiObject, ForcesFixedAndInactive)
{
    Forces f;
    // Label 0 is passive in the evolver, not fixed in the functor.
    EXPECT_FALSE(f.isFixed(0));
    EXPECT_FALSE(f.isFixed(3));
    EXPECT_TRUE(f.isActive(3));
    f.setFixedLabels({4});
    f.setInactiveLabels({5});
    EXPECT_TRUE(f.isFixed(4));
    EXPECT_FALSE(f.isFixed(3));
    EXPECT_FALSE(f.isActive(5));
    EXPECT_TRUE(f.isActive(4));

    auto m = makePair();
    Stencil s(*m);
    s.moveTo(Coord(7, 8, 8), 1);
    EXPECT_FLOAT_EQ(f(Coord(7, 8, 8), 5, s), 0.999f);
    EXPECT_FLOAT_EQ(f(Coord(7, 8, 8), 4, s), 0.0f);

    EXPECT_TRUE(std::isinf(f.stableDt(1.0f)));
    f.setCurvatureWeight(0.5f);
    EXPECT_FLOAT_EQ(f.stableDt(2.0f), 4.0f / 3.0f);
}

TEST_F(TestLevelSetMultiObject, ForcesMissingGrid)
{
    Forces f;
    EXPECT_THROW(f.setPressure(nullptr, 1.0f), ValueError);
    EXPECT_THROW(f.setAdvection(nullptr, 1.0f), ValueError);
}


////////////////////////////////////////
// Evolver

namespace {

// dpsi/dt = s for every non-fixed label (unit normal speed where |grad psi| = 1).
struct ConstantSpeed
{
    float s = 0.0f;
    std::set<int32_t> fixed;
    float operator()(const Coord&, int32_t, const Stencil&) const { return s; }
    bool isFixed(int32_t l) const { return l != 0 && fixed.count(l) != 0; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// Per-label constant dpsi/dt drawn once from a seeded generator in [-1, 1].
struct RandomSpeed
{
    std::map<int32_t, float> speeds;
    RandomSpeed(const std::vector<int32_t>& labels, unsigned seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int32_t l : labels) speeds[l] = dist(gen);
    }
    float operator()(const Coord&, int32_t l, const Stencil&) const
    {
        auto it = speeds.find(l);
        return it == speeds.end() ? 0.0f : it->second;
    }
    bool isFixed(int32_t) const { return false; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// wasInterrupted() returns true from its third call on.
struct CountingInterrupter: public util::NullInterrupter
{
    int calls = 0;
    bool wasInterrupted(int = -1) override { return ++calls >= 3; }
};

// Single sphere of label 1 via fromSignedGrids.
MLS::Ptr
makeSphere(double r, const Vec3d& c = Vec3d(0.0), double dx = 1.0)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c), float(dx));
    return MLS::fromSignedGrids(in);
}

// Zero crossings of a signed grid along active +x/+y/+z voxel edges (world).
std::vector<Vec3d>
zeroCrossings(const FloatGrid& sdf)
{
    std::vector<Vec3d> pts;
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        const float a = *it;
        for (int axis = 0; axis < 3; ++axis) {
            Coord q = p;
            q[axis] += 1;
            float b;
            if (!acc.probeValue(q, b) || ((a < 0.0f) == (b < 0.0f))) continue;
            const double t = double(a) / (double(a) - double(b));
            pts.push_back(sdf.indexToWorld(p.asVec3d() + (q.asVec3d() - p.asVec3d()) * t));
        }
    }
    return pts;
}

double
meanRadius(const std::vector<Vec3d>& pts, const Vec3d& c)
{
    double sum = 0.0;
    for (const Vec3d& p : pts) sum += (p - c).length();
    return pts.empty() ? 0.0 : sum / double(pts.size());
}

Vec3d
centroid(const std::vector<Vec3d>& pts)
{
    Vec3d sum(0.0);
    for (const Vec3d& p : pts) sum += p;
    return pts.empty() ? sum : sum / double(pts.size());
}

// Every active voxel has a voxel of its own label (active or not) among its
// 26 neighbors.
Index64
countOrphans(const MLS& m)
{
    tree::ValueAccessor<const LabelTree> lab(m.labels()->tree());
    Index64 n = 0;
    for (auto it = m.labels()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool found = false;
        for (int i = -1; i <= 1 && !found; ++i)
            for (int j = -1; j <= 1 && !found; ++j)
                for (int k = -1; k <= 1 && !found; ++k) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    found = lab.getValue(p.offsetBy(i, j, k)) == *it;
                }
        if (!found) ++n;
    }
    return n;
}

float
maxActivePhi(const MLS& m)
{
    float mx = 0.0f;
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) mx = std::max(mx, *it);
    return mx;
}

// The interiors {psi_l < 0} of the per-label
// extractSignedDistance grids are pairwise disjoint (up to 1e-3*dx;
// @a overlaps counts violations) and, with the background region, cover the
// band consistently with the labels (@a badCover counts violations).
void
partitionErrors(const MLS& m, Index64& overlaps, Index64& badCover)
{
    const double dx = m.voxelSize();
    const std::vector<int32_t> ls = m.labelSet();
    std::map<int32_t, FloatGrid::Ptr> sdf;
    for (int32_t l : ls) sdf[l] = m.extractSignedDistance(l);
    const float eps = float(-1e-3 * dx);
    overlaps = badCover = 0;
    tree::ValueAccessor<const LabelTree> lab(m.labels()->tree());
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        int inside = 0, deep = 0;
        int32_t which = 0;
        for (int32_t l : ls) {
            const float v = sdf[l]->tree().getValue(p);
            if (v < 0.0f) { ++inside; which = l; }
            if (v < eps) ++deep;
        }
        if (deep > 1) ++overlaps;
        if (inside > 1 || (inside == 1 && lab.getValue(p) != which)
            || (inside == 0 && lab.getValue(p) != 0)) ++badCover;
    }
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, PartitionInvariant)
{
    // 4 labels in a 32^3 checker of 8^3 blocks, background label 0 outside.
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 32; ++i)
            for (int j = 0; j < 32; ++j)
                for (int k = 0; k < 32; ++k) {
                    acc.setValue(Coord(i, j, k), 1 + ((i / 8 + j / 8 + k / 8) % 4));
                }
    }
    auto m = MLS::fromLabelGrid(*labels);
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4}));
    RandomSpeed speed({1, 2, 3, 4}, 20261108u);
    lsmo_frames::FrameEvolver<MLS, RandomSpeed> lsmo(*m, speed);
    for (int n = 0; n < 75; ++n) {
        const float dt = lsmo.step(1.0f);
        ASSERT_GT(dt, 0.0f);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
    }
    EXPECT_EQ(countOrphans(*m), Index64(0));

    ASSERT_FALSE(m->labelSet().empty());
    Index64 overlaps = 0, badCover = 0;
    partitionErrors(*m, overlaps, badCover);
    EXPECT_EQ(overlaps, Index64(0));
    EXPECT_EQ(badCover, Index64(0));
}

namespace {

// Max interface motion argument of the scenario helpers below meaning "keep
// the evolver's default".
constexpr double kDefaultMotion = 0.0;

// Failed assertions recorded so far by the current test.
int
failureCount()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->result()
        ->total_part_count();
}

template<typename EvolverT>
void
applyMotion(EvolverT& e, double motion)
{
    if (motion > kDefaultMotion) e.setMaxInterfaceMotion(motion);
}

// Sphere of radius 8 grown at unit speed to t = 6 (gate: mean radius
// 14 +- 0.5*dx). @return the number of steps.
size_t
runPressureGrowth(double motion)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    ConstantSpeed speed;
    speed.s = -1.0f; // dpsi/dt = -1: the object grows at unit speed
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    const auto start = std::chrono::steady_clock::now();
    const size_t steps = lsmo.advect(0.0, 6.0);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    EXPECT_GE(steps, size_t(6));
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    std::cout << "PressureGrowth: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " (" << 1e3 * secs / double(std::max(steps, size_t(1))) << " ms/step)"
              << " mean radius " << r << " (expected 14)" << std::endl;
    EXPECT_NEAR(r, 8.0 + 6.0, 0.5 * dx);
    return steps;
}

// Sphere of radius 10 under unit curvature flow to t = 30 (gate: r^2 =
// 40 +- 2*sqrt(40)*0.5*dx).
void
runCurvatureFlow(double motion)
{
    auto m = makeSphere(10.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    applyMotion(lsmo, motion);
    const size_t steps = lsmo.advect(0.0, 30.0);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    // r^2 = r0^2 - 2t = 100 - 60
    std::cout << "CurvatureFlow: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " r^2 " << r * r << " (expected 40)" << std::endl;
    EXPECT_NEAR(r * r, 40.0, 2.0 * std::sqrt(40.0) * 0.5 * dx);
}

// Sphere of radius 8 advected by (1, 0, 0) to t = 7.5 (gate: centroid shift
// (7.5, 0, 0) +- 0.5*dx per axis).
void
runUniformAdvection(double motion)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    applyMotion(lsmo, motion);
    const Vec3d c0 = centroid(zeroCrossings(*m->extractSignedDistance(1)));
    const size_t steps = lsmo.advect(0.0, 7.5);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const auto pts = zeroCrossings(*m->extractSignedDistance(1));
    const Vec3d c1 = centroid(pts);
    std::cout << "UniformAdvection: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " shift " << (c1 - c0) << " mean radius " << meanRadius(pts, c1) << std::endl;
    EXPECT_NEAR(c1[0] - c0[0], 7.5, 0.5 * dx);
    EXPECT_NEAR(c1[1] - c0[1], 0.0, 0.5 * dx);
    EXPECT_NEAR(c1[2] - c0[2], 0.0, 0.5 * dx);
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, PressureGrowth)
{
    runPressureGrowth(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, CurvatureFlow)
{
    runCurvatureFlow(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, UniformAdvection)
{
    runUniformAdvection(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, PerLabelAdvection)
{
    // Label 1 uses the shared field (1,0,0), label 2 its own (0,1,0);
    // clearing the override returns label 2 to the shared field.
    auto m = makeTwoSpheres(1.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    forces.setAdvection(2, Vec3SGrid::create(Vec3s(0.0f, 1.0f, 0.0f)), 1.0f);
    EXPECT_THROW(forces.setAdvection(3, Vec3SGrid::ConstPtr(), 1.0f), ValueError);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    auto center = [&](int32_t l) { return centroid(zeroCrossings(*m->extractSignedDistance(l))); };
    const Vec3d a0 = center(1), b0 = center(2);
    lsmo.advect(0.0, 5.0);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const Vec3d a1 = center(1), b1 = center(2);
    std::cout << "PerLabelAdvection: label 1 shift " << (a1 - a0) << ", label 2 shift "
              << (b1 - b0) << std::endl;
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(a1[k] - a0[k], k == 0 ? 5.0 : 0.0, 0.5 * dx) << "axis " << k;
        EXPECT_NEAR(b1[k] - b0[k], k == 1 ? 5.0 : 0.0, 0.5 * dx) << "axis " << k;
    }
    forces.clearLabelAdvection(2);
    lsmo.advect(5.0, 7.0);
    const Vec3d b2 = center(2);
    EXPECT_NEAR(b2[0] - b1[0], 2.0, 0.5 * dx);
    EXPECT_NEAR(b2[1] - b1[1], 0.0, 0.5 * dx);
}

TEST_F(TestLevelSetMultiObject, KinematicLabelBasics)
{
    // A kinematic sphere (label 2, radius 4) translating along +x at unit
    // speed through label 0 and through a non-protected block (label 1):
    // its region is exactly the moved ball at every chunk (it never loses
    // voxels), the block is pushed aside (none of its voxels inside the
    // ball, the block persists), validate and the partition hold.
    const double r = 4.0;
    const Vec3d c0(-10.0, 15.0, 15.0);
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(6, 9, 9), Coord(24, 21, 21)).begin(); p; ++p) acc.setValue(*p, 1);
        for (auto p = CoordBBox(Coord::round(c0) - Coord(5), Coord::round(c0) + Coord(5)).begin(); p; ++p) {
            if (((*p).asVec3d() - c0).length() < r) acc.setValue(*p, 2);
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    Forces forces; // no speeds: the block moves only where the sphere pushes it
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    EXPECT_THROW(lsmo.setKinematicLabel(0, FloatGrid::create(), [](double) {
        return math::Mat4d::identity(); }), ValueError);
    EXPECT_THROW(lsmo.setKinematicLabel(2, FloatGrid::ConstPtr(), [](double) {
        return math::Mat4d::identity(); }), ValueError);
    EXPECT_THROW(lsmo.setKinematicLabel(2, FloatGrid::create(), nullptr), ValueError);
    auto motion = [](double t) {
        math::Mat4d mm = math::Mat4d::identity();
        mm.setTranslation(Vec3d(t, 0.0, 0.0));
        return mm;
    };
    lsmo.setKinematicLabel(2, tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c0), 1.0f),
        motion);
    const CoordBBox domain(Coord(-20, 0, 0), Coord(40, 30, 30));
    auto count = [&](int32_t l) {
        Index64 n = 0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) n += acc.getValue(*p) == l;
        return n;
    };
    const Index64 block0 = count(1);
    Index64 minBlock = block0;
    for (double t = 0.0; t < 44.0 - 1e-9; t += 2.0) {
        lsmo.advect(t, t + 2.0);
        const Vec3d c = c0 + Vec3d(t + 2.0, 0.0, 0.0);
        Index64 missing = 0, extra = 0, blockInside = 0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            const double d = ((*p).asVec3d() - c).length() - r;
            const int32_t l = acc.getValue(*p);
            if (d < -0.1 && l != 2) ++missing;
            if (d > 0.1 && l == 2) ++extra;
            if (d < 0.0 && l == 1) ++blockInside;
        }
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "t " << t + 2.0 << ": " << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*m, overlaps, badCover);
        EXPECT_EQ(missing, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(extra, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(blockInside, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(overlaps, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(badCover, Index64(0)) << "t " << t + 2.0;
        minBlock = std::min(minBlock, count(1));
    }
    const Index64 block1 = count(1);
    std::cout << "KinematicLabelBasics: block voxels " << block0 << " -> min " << minBlock
              << " -> " << block1 << ", sphere voxels " << count(2) << std::endl;
    EXPECT_LT(minBlock, block0);  // the sphere displaced block voxels
    EXPECT_GT(block1, block0 / 2); // the block persists
    EXPECT_NEAR(lsmo.time(), 44.0, 1e-6);
}

TEST_F(TestLevelSetMultiObject, KinematicFixedWall)
{
    // A kinematic sphere (label 2) pushed through a fixed wall (label 3)
    // never takes or changes wall voxels: the wall is bit-identical.
    const double r = 4.0;
    const Vec3d c0(-8.0, 15.0, 15.0);
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(4, 8, 8), Coord(6, 22, 22)).begin(); p; ++p) acc.setValue(*p, 3);
        for (auto p = CoordBBox(Coord::round(c0) - Coord(5), Coord::round(c0) + Coord(5)).begin(); p; ++p) {
            if (((*p).asVec3d() - c0).length() < r) acc.setValue(*p, 2);
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    Forces forces;
    forces.setFixedLabels({3});
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    lsmo.setKinematicLabel(2, tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c0), 1.0f),
        [](double t) {
            math::Mat4d mm = math::Mat4d::identity();
            mm.setTranslation(Vec3d(t, 0.0, 0.0));
            return mm;
        });
    struct Snap { int32_t label; bool on; float phi; };
    const CoordBBox box(Coord(-20, 0, 0), Coord(30, 30, 30));
    auto wall = [&]() {
        std::map<Coord, Snap> w;
        for (auto p = box.begin(); p; ++p) {
            if (m->labels()->tree().getValue(*p) != 3) continue;
            w[*p] = Snap{3, m->phi()->tree().isValueOn(*p), m->phi()->tree().getValue(*p)};
        }
        return w;
    };
    const auto wall0 = wall();
    ASSERT_FALSE(wall0.empty());
    lsmo.advect(0.0, 20.0); // the sphere reaches and presses into the wall
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const auto wall1 = wall();
    EXPECT_EQ(wall1.size(), wall0.size());
    Index64 diff = 0;
    for (const auto& kv : wall0) {
        auto it = wall1.find(kv.first);
        if (it == wall1.end() || it->second.on != kv.second.on
            || std::memcmp(&it->second.phi, &kv.second.phi, sizeof(float)) != 0) ++diff;
    }
    EXPECT_EQ(diff, Index64(0));
    // kinematic and topology-preserving labels exclude each other, and
    // label 0 cannot be protected together with kinematic labels
    EXPECT_THROW(lsmo.setTopologyPreservingLabels({2}), ValueError);
    EXPECT_THROW(lsmo.setTopologyPreservingLabels({0}), ValueError);
    lsmo.setTopologyPreservingLabels({1});
    EXPECT_THROW(lsmo.setKinematicLabel(1, tools::createLevelSetSphere<FloatGrid>(1.0f,
        Vec3f(0.0f), 1.0f), [](double) { return math::Mat4d::identity(); }), ValueError);
    // the phi floor: settable, only in effect with protected labels
    EXPECT_THROW(lsmo.setMinimumPhi(-1.0), ValueError);
    lsmo.setMinimumPhi(5e-3);
    EXPECT_DOUBLE_EQ(lsmo.minimumPhi(), 5e-3);
    EXPECT_DOUBLE_EQ(lsmo.phiFloor(), 5e-3 * m->voxelSize());
    lsmo.setTopologyPreservingLabels({});
    EXPECT_DOUBLE_EQ(lsmo.phiFloor(), 0.0);
    EXPECT_TRUE(lsmo.validateTopologyFloor());
}

namespace {

// Smooth test fields evaluated at world positions, on a box of voxels of
// @a xform (index range [-26, 26]^3, background zero).
FloatGrid::Ptr
makePressureField(const math::Transform::Ptr& xform, double dx)
{
    FloatGrid::Ptr g = FloatGrid::create(0.0f);
    g->setTransform(xform);
    auto acc = g->getAccessor();
    for (int i = -26; i <= 26; ++i)
        for (int j = -26; j <= 26; ++j)
            for (int k = -26; k <= 26; ++k) {
                const Vec3d w = xform->indexToWorld(Coord(i, j, k)) / dx;
                acc.setValue(Coord(i, j, k),
                    float(0.8 + 0.5 * std::sin(0.3 * w.x()) + 0.3 * std::cos(0.2 * w.y())
                          - 0.1 * w.z() / 26.0));
            }
    return g;
}

Vec3SGrid::Ptr
makeVelocityField(const math::Transform::Ptr& xform, double dx)
{
    Vec3SGrid::Ptr g = Vec3SGrid::create(Vec3f(0.0f));
    g->setTransform(xform);
    auto acc = g->getAccessor();
    for (int i = -26; i <= 26; ++i)
        for (int j = -26; j <= 26; ++j)
            for (int k = -26; k <= 26; ++k) {
                const Vec3d w = xform->indexToWorld(Coord(i, j, k)) / dx;
                acc.setValue(Coord(i, j, k),
                    Vec3f(float(0.8 + 0.2 * std::sin(0.2 * w.y())),
                          float(0.3 * std::cos(0.15 * w.x())),
                          float(0.1 + 0.05 * std::sin(0.1 * w.z()))));
            }
    return g;
}

// Evolve a sphere under pressure + advection fields living on a transform
// translated by @a shift voxels from the pair's; @a force selects the
// BoxSampler path.
MLS::Ptr
runFieldScenario(const Vec3d& shift, bool force, double dx = 1.0, bool extension = false)
{
    auto m = makeSphere(8.0 * dx, Vec3d(0.0), dx);
    math::Transform::Ptr xf = m->phi()->transform().copy();
    xf->postTranslate(shift * m->voxelSize());
    Forces forces;
    forces.setForceSampling(force);
    forces.setSpeedExtension(extension);
    forces.setPressure(makePressureField(xf, dx), 1.0f);
    forces.setAdvection(makeVelocityField(xf, dx), 1.0f);
    forces.setCurvatureWeight(0.1f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    lsmo.advect(0.0, 3.0 * dx);
    return m;
}

Index64
bitDifferences(const MLS& a, const MLS& b)
{
    Index64 diff = 0;
    const FloatTree& pa = a.phi()->tree();
    const FloatTree& pb = b.phi()->tree();
    if (!a.labels()->tree().hasSameTopology(b.labels()->tree())) ++diff;
    for (auto it = pa.cbeginValueOn(); it; ++it) {
        const float v = pb.getValue(it.getCoord());
        if (std::memcmp(&v, &*it, sizeof(float)) != 0) ++diff;
    }
    for (auto it = a.labels()->tree().cbeginValueAll(); it; ++it) {
        if (b.labels()->tree().getValue(it.getCoord()) != *it) ++diff;
    }
    return diff;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, FieldDirectReadMatchesSampler)
{
    // Fields on the pair's transform are read directly; the result is
    // bit-identical to the BoxSampler path. Fields on a translated transform
    // keep using the sampler.
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    auto direct = runFieldScenario(Vec3d(0.0), false);
    const auto t1 = Clock::now();
    auto sampled = runFieldScenario(Vec3d(0.0), true);
    const auto t2 = Clock::now();
    std::cout << "FieldDirectReadMatchesSampler: direct "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, sampled "
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
    ASSERT_FALSE(direct->phi()->tree().empty());
    EXPECT_EQ(bitDifferences(*direct, *sampled), Index64(0));

    const Vec3d shift(0.5, 0.25, 0.0);
    auto mis = runFieldScenario(shift, false);
    auto misSampled = runFieldScenario(shift, true);
    EXPECT_EQ(bitDifferences(*mis, *misSampled), Index64(0));
    // the translated field really takes the sampler path: it differs from
    // the aligned (direct) result
    EXPECT_GT(bitDifferences(*mis, *direct), Index64(0));

    // non-unit voxel size: worldToIndex(indexToWorld(c)) must round-trip
    auto direct1 = runFieldScenario(Vec3d(0.0), false, 0.1);
    auto sampled1 = runFieldScenario(Vec3d(0.0), true, 0.1);
    ASSERT_FALSE(direct1->phi()->tree().empty());
    EXPECT_EQ(bitDifferences(*direct1, *sampled1), Index64(0));

    // speed extension (closest-point samples always interpolate): the
    // force-sampling hook changes nothing, and the result differs from
    // voxel-center sampling
    const auto t3 = Clock::now();
    auto ext = runFieldScenario(Vec3d(0.0), false, 1.0, true);
    const auto t4 = Clock::now();
    std::cout << "FieldDirectReadMatchesSampler: speed extension "
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms" << std::endl;
    auto extSampled = runFieldScenario(Vec3d(0.0), true, 1.0, true);
    EXPECT_EQ(bitDifferences(*ext, *extSampled), Index64(0));
    EXPECT_GT(bitDifferences(*ext, *direct), Index64(0));
}

namespace {

// Sphere 1 grows into the fixed sphere 2 over 30 steps of at most 0.75
// (gate: no voxel of label 2 changes; the labels end in contact).
void
runFixedLabels(double motion)
{
    auto m = makeTwoSpheres(1.0);
    std::map<Coord, float> before;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it == 2) before[it.getCoord()] = m->phi()->tree().getValue(it.getCoord());
    }
    ASSERT_FALSE(before.empty());
    ConstantSpeed speed;
    speed.s = -1.0f;
    speed.fixed = {2};
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    for (int n = 0; n < 30; ++n) {
        // dt = min(0.75, motion*cfl): at the default motion 0.5 sphere 1
        // grows ~15 voxels, reaching the wall after ~20 steps
        lsmo.step(0.75f);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
    }
    Index64 changed = 0;
    for (const auto& kv : before) {
        const float phi = m->phi()->tree().getValue(kv.first);
        if (!m->labels()->tree().isValueOn(kv.first)
            || m->labels()->tree().getValue(kv.first) != 2
            || std::memcmp(&phi, &kv.second, sizeof(float)) != 0) ++changed;
    }
    EXPECT_EQ(changed, Index64(0));
    // the moving label is in contact with the fixed one
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    Index64 contact = 0;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it != 1) continue;
        for (int f = 0; f < 6; ++f) contact += m->labels()->tree().getValue(it.getCoord() + offs[f]) == 2;
    }
    EXPECT_GT(contact, Index64(0));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, FixedLabels)
{
    runFixedLabels(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, BandMaintenance)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    ConstantSpeed speed;
    speed.s = -1.0f;
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
    double t = 0.0;
    int n = 0;
    while (t < 6.0) {
        t += lsmo.step(float(6.0 - t));
        ++n;
        ASSERT_TRUE(m->phi()->tree().hasSameTopology(m->labels()->tree())) << "step " << n;
        ASSERT_LE(maxActivePhi(*m), float((m->halfWidth() + 1) * dx)) << "step " << n;
        // inactive phi is the background
        for (auto it = m->phi()->tree().cbeginValueOff(); it; ++it) {
            ASSERT_EQ(*it, m->phi()->background()) << "step " << n;
        }
    }
    EXPECT_GE(n, 6);
}

TEST_F(TestLevelSetMultiObject, EmptyStep)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    {
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 8; ++k) {
                    pa.setValue(Coord(i, j, k), 1.0f);
                    la.setValue(Coord(i, j, k), 0);
                }
    }
    MLS m(phi, lab, 3);
    const FloatTree phiBefore(m.phi()->tree());
    const LabelTree labBefore(m.labels()->tree());
    ConstantSpeed speed;
    speed.s = -1.0f;
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(m, speed);
    EXPECT_EQ(lsmo.step(0.5f), 0.5f);
    EXPECT_TRUE(m.phi()->tree().hasSameTopology(phiBefore));
    EXPECT_TRUE(m.labels()->tree().hasSameTopology(labBefore));
    for (auto it = phiBefore.cbeginValueOn(); it; ++it) {
        EXPECT_EQ(m.phi()->tree().getValue(it.getCoord()), *it);
        EXPECT_EQ(m.labels()->tree().getValue(it.getCoord()), 0);
    }
}

TEST_F(TestLevelSetMultiObject, StepPeakMemory)
{
    // Two spheres of radius 15 at (+-16, 0, 0) in a 64^3 box (dx = 1, halfWidth 3),
    // both growing into contact. The evolver's own working memory during a step
    // (auxiliary leaf buffers plus pooled per-leaf scratch) stays within 1.3x the
    // memory of the pair (a deep copy of both trees alone would be about 1x, on
    // top of the candidate lists), also once the pooled scratch has been reused
    // over many steps of a growing band. Pair bytes are taken before each step
    // (tree.memUsage, an approximation of the allocated bytes). TVD-RK2 keeps
    // the time-n state for its second stage, so it may also use the phi
    // tree's bytes on top.
    for (math::TemporalIntegrationScheme scheme : {math::TVD_RK1, math::TVD_RK2}) {
        const std::string name = scheme == math::TVD_RK1 ? "TVD_RK1" : "TVD_RK2";
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(15.0f, Vec3f(-16.0f, 0.0f, 0.0f), 1.0f);
        in[2] = tools::createLevelSetSphere<FloatGrid>(15.0f, Vec3f(16.0f, 0.0f, 0.0f), 1.0f);
        auto m = MLS::fromSignedGrids(in, 3);
        ConstantSpeed speed;
        speed.s = -1.0f;
        lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
        lsmo.setTemporalScheme(scheme);
        EXPECT_EQ(lsmo.scratchMemUsage(), size_t(0));
        for (int n = 0; n < 16; ++n) {
            const size_t phiBytes = size_t(m->phi()->memUsage());
            const size_t pairBytes = phiBytes + size_t(m->labels()->memUsage());
            ASSERT_GT(lsmo.step(1.0f), 0.0f);
            const size_t scratch = lsmo.scratchMemUsage();
            const double bound = 1.3 * double(pairBytes)
                + (scheme == math::TVD_RK2 ? double(phiBytes) : 0.0);
            std::cout << "StepPeakMemory " << name << ": step " << n << " pair " << pairBytes
                      << " B, phi " << phiBytes << " B, scratch " << scratch << " B ("
                      << double(scratch) / double(pairBytes) << "x pair, bound "
                      << bound / double(pairBytes) << "x)" << std::endl;
            EXPECT_GT(scratch, size_t(0)) << name << " step " << n;
            EXPECT_LE(double(scratch), bound) << name << " step " << n;
        }
        std::string msg;
        EXPECT_TRUE(m->validate(&msg)) << msg;
    }
}

TEST_F(TestLevelSetMultiObject, InterruptMidAdvect)
{
    auto m = makeSphere(8.0);
    ConstantSpeed speed;
    speed.s = -1.0f;
    CountingInterrupter interrupter;
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed, CountingInterrupter> lsmo(*m, speed, &interrupter);
    // uninterrupted, t = 15 needs at least 15 steps (dt <= cfl*dx)
    const size_t steps = lsmo.advect(0.0, 15.0);
    EXPECT_LT(steps, size_t(15));
    EXPECT_EQ(steps, size_t(2));
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, SplitStepInterrupt)
{
    // Pure curvature flow: one split step would cover t = 30 (no hyperbolic
    // speed), in parabolic substeps of dx^2/6. The interrupter fires on its
    // third check: advect's check before the step, then after each of the
    // first two substeps.
    auto m = makeSphere(10.0);
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    CountingInterrupter interrupter;
    lsmo_frames::FrameEvolver<MLS, Forces, CountingInterrupter> lsmo(*m, forces, &interrupter);
    const size_t steps = lsmo.advect(0.0, 30.0);
    EXPECT_EQ(steps, size_t(1));
    EXPECT_EQ(interrupter.calls, 3);
    EXPECT_GT(lsmo.time(), 0.0);
    EXPECT_LT(lsmo.time(), 30.0);
    EXPECT_NEAR(lsmo.time(), 2.0 / 6.0, 1e-5); // two substeps of dx^2/6
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, ThinFeatureVanishes)
{
    // 1-voxel-thick slab of label 1 inside a 16^3 block of label 2
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k) acc.setValue(Coord(i, j, k), i == 8 ? 1 : 2);
    }
    auto m = MLS::fromLabelGrid(*labels);
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    forces.setPressure(FloatGrid::create(-1.0f), 1.0f); // inward
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    for (int n = 0; n < 45; ++n) {
        lsmo.step(1.0f);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
    }
    for (auto it = m->phi()->tree().cbeginValueAll(); it; ++it) {
        ASSERT_TRUE(std::isfinite(*it)) << "NaN/inf phi at " << it.getCoord();
    }
    EXPECT_EQ(countOrphans(*m), Index64(0));
    const auto ls = m->labelSet();
    EXPECT_EQ(std::count(ls.begin(), ls.end(), 1), 0) << "the slab did not vanish";
}

TEST_F(TestLevelSetMultiObject, EvolverSerialMatchesParallel)
{
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                for (int k = 0; k < 24; ++k) acc.setValue(Coord(i, j, k), 1 + ((i / 8 + j / 8 + k / 8) % 3));
    }
    auto a = MLS::fromLabelGrid(*labels);
    auto b = MLS::fromLabelGrid(*labels);
    RandomSpeed speed({1, 2, 3}, 7u);
    lsmo_frames::FrameEvolver<MLS, RandomSpeed> serial(*a, speed), parallel(*b, speed);
    serial.setGrainSize(0);
    for (int n = 0; n < 15; ++n) {
        EXPECT_EQ(serial.step(1.0f), parallel.step(1.0f));
    }
    ASSERT_TRUE(a->phi()->tree().hasSameTopology(b->phi()->tree()));
    Index64 diff = 0;
    for (auto it = a->phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        diff += (*it != b->phi()->tree().getValue(p))
            || (a->labels()->tree().getValue(p) != b->labels()->tree().getValue(p));
    }
    EXPECT_EQ(diff, Index64(0));
}

namespace {

// Per-label constant dpsi/dt.
struct LabelSpeed
{
    std::map<int32_t, float> speeds;
    std::set<int32_t> fixed;
    float operator()(const Coord&, int32_t l, const Stencil&) const
    {
        auto it = speeds.find(l);
        return it == speeds.end() ? 0.0f : it->second;
    }
    bool isFixed(int32_t l) const { return l != 0 && fixed.count(l) != 0; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// Review probe of ContactNoCascade; @a motion as for runPressureGrowth.
void
runContactNoCascade(double motion)
{
    // Review probe: label 1 slab x < 10, label 2 block x in [10,13), y < 3,
    // in contact; label 2 is faster. Along the contact shell (10, y, 12)
    // label 2 may advance at most one voxel per step (the evolver caps dt so
    // interfaces move at most maxInterfaceMotion()*cfl*dx per step: half a
    // voxel by default).
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 13; ++i)
            for (int j = 0; j < 24; ++j)
                for (int k = 0; k < 24; ++k) {
                    if (i < 10) acc.setValue(Coord(i, j, k), 1);
                    else if (j < 3) acc.setValue(Coord(i, j, k), 2);
                }
    }
    auto m = MLS::fromLabelGrid(*labels);
    LabelSpeed speed;
    speed.speeds = {{1, -0.5f}, {2, -1.0f}};
    lsmo_frames::FrameEvolver<MLS, LabelSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    const double cap = lsmo.maxInterfaceMotion();
    auto extent = [&]() {
        int maxj = -1;
        for (int j = 0; j < 24; ++j) {
            const Coord c(10, j, 12);
            if (m->labels()->tree().isValueOn(c) && m->labels()->tree().getValue(c) == 2) maxj = j;
        }
        return maxj;
    };
    auto count = [&]() {
        Index64 n = 0;
        for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) n += *it == 2;
        return n;
    };
    const int y0 = extent();
    ASSERT_EQ(y0, 2);
    std::vector<Index64> counts{count()};
    std::vector<int> extents{y0};
    const double cfl = 0.999;
    // 1.5 x the time of three full steps: four full steps and a half step
    const double tEnd = 1.5 * 3.0 * cap * cfl;
    double t = 0.0;
    int nSteps = 0;
    while (t < tEnd - 1e-6) {
        const double expected = std::min(cap * cfl, tEnd - t);
        const float dt = lsmo.step(float(tEnd - t));
        // max|F| = 1: at most the motion cap (R24)
        EXPECT_FLOAT_EQ(dt, float(expected)) << "step " << nSteps + 1;
        t += dt;
        ++nSteps;
        counts.push_back(count());
        extents.push_back(extent());
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
    }
    ASSERT_EQ(nSteps, 5);
    std::cout << "ContactNoCascade: motion " << cap << " t " << t << ", extent";
    for (int e : extents) std::cout << " " << e;
    std::cout << ", label-2 voxels";
    for (Index64 c : counts) std::cout << " " << c;
    std::cout << std::endl;
    for (int s = 1; s <= nSteps; ++s) {
        EXPECT_LE(extents[s] - extents[s - 1], int(std::floor(cfl + 0.5))) << "step " << s;
    }
    // cumulative: the front (speed 1) has moved t; one voxel of slack at the
    // junction with label 1, where psi_2 is only the pair lower bound
    EXPECT_LE(extents[nSteps] - extents[0], int(std::floor(t + 1.0 + 1e-6)));
    // No cascade anywhere: after the run every label-2 voxel (active or not)
    // is within t + dx of a voxel of the initial block
    // x in [10,13), y in [0,3), z in [0,24).
    Index64 far = 0;
    const double bound = t + 1.0;
    for (auto it = m->labels()->tree().cbeginValueAll(); it; ++it) {
        if (*it != 2) continue;
        CoordBBox box;
        it.getBoundingBox(box);
        for (auto p = box.begin(); p; ++p) {
            const Coord c = *p;
            const double dxv = std::max({10 - c[0], c[0] - 12, 0});
            const double dyv = std::max({0 - c[1], c[1] - 2, 0});
            const double dzv = std::max({0 - c[2], c[2] - 23, 0});
            if (std::sqrt(dxv * dxv + dyv * dyv + dzv * dzv) > bound + 1e-6) ++far;
        }
    }
    EXPECT_EQ(far, Index64(0));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, ContactNoCascade)
{
    runContactNoCascade(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, WallRetreat)
{
    // Label 2 (fixed) for x < 8, label 1 for x in [8,16); label 1 retreats
    // from the wall: its voxels next to the wall become passive label 0.
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k) acc.setValue(Coord(i, j, k), i < 8 ? 2 : 1);
    }
    auto m = MLS::fromLabelGrid(*labels);
    LabelSpeed speed;
    speed.speeds = {{1, 1.0f}};
    speed.fixed = {2};
    lsmo_frames::FrameEvolver<MLS, LabelSpeed> lsmo(*m, speed);
    const Coord wallSide(7, 8, 8), retreat(8, 8, 8);
    const float wallPhi = m->phi()->tree().getValue(wallSide);
    ASSERT_EQ(m->labels()->tree().getValue(retreat), 1);
    // three steps of dt = 0.5*cfl (half-voxel step cap) move the label-1 boundary ~1.5 voxels
    for (int n = 0; n < 3; ++n) lsmo.step(1.0f);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labels()->tree().getValue(retreat), 0);
    EXPECT_EQ(m->labels()->tree().getValue(wallSide), 2);
    EXPECT_EQ(m->phi()->tree().getValue(wallSide), wallPhi);
}


////////////////////////////////////////
// Single-object equivalence with OpenVDB's single
// level set path (LevelSetAdvection / LevelSetFilter, FIRST_BIAS + TVD_RK1).

namespace {

struct ZeroSetDiff
{
    double symDiff = 0.0;     // sub-voxel symmetric difference volume (world)
    double symDiffSign = 0.0; // voxel count with differing signs, times dx^3
    double hausdorff = 0.0;   // Hausdorff distance of the zero sets (world)
};

// One-directional zero-set distance: max |psi_b| at the volumeToMesh
// points (isovalue 0) of @a a, sampled with BoxSampler.
double
zeroSetDistance(const FloatGrid& a, const FloatGrid& b)
{
    std::vector<Vec3s> pts;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(a, pts, quads, 0.0);
    double d = 0.0;
    for (const Vec3s& p : pts) {
        const Vec3d idx = b.transform().worldToIndex(Vec3d(p));
        d = std::max(d, double(std::abs(tools::BoxSampler::sample(b.tree(), idx))));
    }
    return d;
}

// Compares the zero sets of two signed distance grids on the same transform.
// Symmetric difference: over the union of both active bands, each voxel
// contributes |H(a) - H(b)|*dx^3 with the sub-voxel inside fraction
// H(psi) = clamp(0.5 - psi/dx, 0, 1) (asserted); symDiffSign is the plain
// count of voxels whose signs differ (reported only). Hausdorff: max of the
// two one-directional zero-set distances.
ZeroSetDiff
compareZeroSets(const FloatGrid& a, const FloatGrid& b)
{
    const double dx = a.voxelSize()[0], dx3 = dx * dx * dx;
    auto H = [dx](float v) { return std::min(1.0, std::max(0.0, 0.5 - double(v) / dx)); };
    tree::ValueAccessor<const FloatTree> aa(a.tree()), ba(b.tree());
    ZeroSetDiff r;
    auto visit = [&](const Coord& c) {
        const float va = aa.getValue(c), vb = ba.getValue(c);
        r.symDiff += std::abs(H(va) - H(vb)) * dx3;
        if ((va < 0.0f) != (vb < 0.0f)) r.symDiffSign += dx3;
    };
    for (auto it = a.tree().cbeginValueOn(); it; ++it) visit(it.getCoord());
    for (auto it = b.tree().cbeginValueOn(); it; ++it) {
        if (!aa.isValueOn(it.getCoord())) visit(it.getCoord());
    }
    r.hausdorff = std::max(zeroSetDistance(a, b), zeroSetDistance(b, a));
    return r;
}

// Fixture: sphere r = 12 dx, centered off the rotation axis so the rotational
// field moves it.
const double kEqRadius = 12.0;
const Vec3d kEqCenter(16.0, 0.0, 0.0);

// v = 0.05*(-y, x, 0) over a box covering the swept annulus (plus band).
Vec3SGrid::Ptr
makeRotation()
{
    return makeVelocity(CoordBBox(Coord(-36, -36, -18), Coord(36, 36, 18)), 1.0,
        [](const Vec3d& p) { return Vec3s(float(-0.05 * p[1]), float(0.05 * p[0]), 0.0f); });
}

void
referenceAdvect(FloatGrid& ref, const Vec3SGrid& vel, double t0, double t1,
    math::BiasedGradientScheme spatial = math::FIRST_BIAS,
    math::TemporalIntegrationScheme temporal = math::TVD_RK1)
{
    using Field = tools::DiscreteField<Vec3SGrid>;
    Field field(vel);
    tools::LevelSetAdvection<FloatGrid, Field> adv(ref, field);
    adv.setSpatialScheme(spatial);
    adv.setTemporalScheme(temporal);
    adv.advect(float(t0), float(t1));
}

void
referenceCurvature(FloatGrid& ref)
{
    tools::LevelSetFilter<FloatGrid> filter(ref);
    filter.meanCurvature(); // dt = dx^2/3
}

void
checkEquivalence(const char* name, const MLS& m, const FloatGrid& ref, size_t steps,
    double angle, const FloatGrid* analytic = nullptr, double exactRadius = -1.0,
    double radiusSlack = 0.0)
{
    std::string msg;
    ASSERT_TRUE(m.validate(&msg)) << msg;
    const double dx = m.voxelSize();
    FloatGrid::Ptr sdf = m.extractSignedDistance(1);
    // the center, rotated by @a angle about z, is inside
    const Vec3d c(kEqCenter[0] * std::cos(angle), kEqCenter[0] * std::sin(angle), 0.0);
    ASSERT_LT(sdf->tree().getValue(sdf->transform().worldToIndexCellCentered(c)), 0.0f);
    const double vol = tools::levelSetVolume(ref);
    const ZeroSetDiff d = compareZeroSets(*sdf, ref);
    std::cout << name << ": MOGAC steps " << steps << ", reference volume " << vol
              << ", MOGAC volume " << tools::levelSetVolume(*sdf)
              << ", symmetric difference " << d.symDiff << " (" << 100.0 * d.symDiff / vol
              << "%), sign-count " << d.symDiffSign << " (" << 100.0 * d.symDiffSign / vol
              << "%), Hausdorff " << d.hausdorff / dx << " dx" << std::endl;
    EXPECT_LT(d.hausdorff, 1.0 * dx);
    if (exactRadius > 0.0) {
        // Mean zero-crossing radius about the exact center @a c; MOGAC's
        // radius error at most max(1.1 x the reference's, the reference's +
        // @a radiusSlack, 0.25*dx).
        const double rm = meanRadius(zeroCrossings(*sdf), c);
        const double rr = meanRadius(zeroCrossings(ref), c);
        const double em = std::abs(rm - exactRadius), er = std::abs(rr - exactRadius);
        std::cout << name << ": exact radius " << exactRadius << ", MOGAC mean radius " << rm
                  << " (error " << em / dx << " dx), reference mean radius " << rr
                  << " (error " << er / dx << " dx)" << std::endl;
        EXPECT_LE(em, std::max({1.1 * er, er + radiusSlack, 0.25 * dx}));
        return;
    }
    ASSERT_TRUE(analytic) << "give an exact grid or an exact radius";
    // Both first-order schemes diffuse, by different amounts, so
    // MOGAC must be at least as close to the exact solution as the reference
    // (10% slack) instead of within 1% of the reference.
    const ZeroSetDiff dm = compareZeroSets(*sdf, *analytic), dr = compareZeroSets(ref, *analytic);
    std::cout << name << ": vs analytic: MOGAC symmetric difference " << dm.symDiff
              << " Hausdorff " << dm.hausdorff / dx << " dx; reference " << dr.symDiff
              << " Hausdorff " << dr.hausdorff / dx << " dx" << std::endl;
    EXPECT_LE(dm.symDiff, 1.1 * dr.symDiff);
}

// The schemes of the first-order reference (LevelSetAdvection with
// FIRST_BIAS / TVD_RK1), and, given @a forces, its field sampling at the
// voxel centers (speed extension off) for the equivalence tests. The evolver
// holds the forces by reference, so the extension may be switched off after
// constructing it.
template<typename EvolverT>
void
pinFirstOrder(EvolverT& e, Forces* forces = nullptr)
{
    e.setSpatialScheme(math::FIRST_BIAS);
    e.setTemporalScheme(math::TVD_RK1);
    if (forces) forces->setSpeedExtension(false);
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, SingleObjectAdvection)
{
    Vec3SGrid::Ptr vel = makeRotation();
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    referenceAdvect(*ref, *vel, 0.0, 30.0);

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setAdvection(vel, 1.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    pinFirstOrder(lsmo, &forces); // compared against the first-order reference
    const size_t steps = lsmo.advect(0.0, 30.0);
    // exact solution: the sphere rotated by 0.05*30 = 1.5 radians about z
    const double angle = 0.05 * 30.0;
    FloatGrid::Ptr exact = tools::createLevelSetSphere<FloatGrid>(float(kEqRadius),
        Vec3f(float(kEqCenter[0] * std::cos(angle)), float(kEqCenter[0] * std::sin(angle)), 0.0f),
        1.0f);
    checkEquivalence("SingleObjectAdvection", *m, *ref, steps, angle, exact.get());
}

TEST_F(TestLevelSetMultiObject, SingleObjectCurvature)
{
    // 90 iterations of dt = dx^2/3: total t = 30
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    for (int n = 0; n < 90; ++n) referenceCurvature(*ref);

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    pinFirstOrder(lsmo, &forces); // compared against the first-order reference
    const size_t steps = lsmo.advect(0.0, 30.0);
    // exact: r^2 = r0^2 - 2t at the unmoved center
    checkEquivalence("SingleObjectCurvature", *m, *ref, steps, 0.0, nullptr,
        std::sqrt(kEqRadius * kEqRadius - 2.0 * 30.0));
}

TEST_F(TestLevelSetMultiObject, SingleObjectCombined)
{
    // Reference: 45 x (advection over dx^2/3, one meanCurvature iteration),
    // total t = 15.
    Vec3SGrid::Ptr vel = makeRotation();
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    const double h = 1.0 / 3.0;
    for (int n = 0; n < 45; ++n) {
        referenceAdvect(*ref, *vel, n * h, (n + 1) * h);
        referenceCurvature(*ref);
    }

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setAdvection(vel, 1.0f);
    forces.setCurvatureWeight(1.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    pinFirstOrder(lsmo, &forces); // compared against the first-order reference
    EXPECT_TRUE(lsmo.splitting()); // the default for a split speed functor
    const size_t steps = lsmo.advect(0.0, 15.0);
    // exact: r^2 = r0^2 - 2t about the center rotated by 0.05*15 radians.
    // 0.1*dx extra slack over the reference's radius error. Strang splitting
    // lets the advection run at its own CFL with the curvature substepped at
    // its limit dx^2/(6w); unsplit, that limit caps every step and forces the
    // advection to CFL ~0.13, where first-order upwinding is most diffusive.
    checkEquivalence("SingleObjectCombined", *m, *ref, steps, 0.05 * 15.0, nullptr,
        std::sqrt(kEqRadius * kEqRadius - 2.0 * 15.0), 0.1 * m->voxelSize());
}

TEST_F(TestLevelSetMultiObject, SchemeDefaultsAndSetters)
{
    auto m = makeSphere(8.0);
    Forces forces;
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    EXPECT_EQ(lsmo.spatialScheme(), math::HJWENO5_BIAS);
    EXPECT_EQ(lsmo.temporalScheme(), math::TVD_RK2);
    for (math::BiasedGradientScheme s : {math::SECOND_BIAS, math::THIRD_BIAS, math::WENO5_BIAS}) {
        EXPECT_THROW(lsmo.setSpatialScheme(s), ValueError) << s;
    }
    EXPECT_THROW(lsmo.setTemporalScheme(math::TVD_RK3), ValueError);
    lsmo.setSpatialScheme(math::FIRST_BIAS);
    lsmo.setTemporalScheme(math::TVD_RK1);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS);
    EXPECT_EQ(lsmo.temporalScheme(), math::TVD_RK1);
    // a stencil made without a scheme is first order
    tools::LabelAwareStencil<MLS> s(*m);
    EXPECT_EQ(s.spatialScheme(), math::FIRST_BIAS);
    EXPECT_THROW(tools::LabelAwareStencil<MLS>(*m, math::WENO5_BIAS), ValueError);
}

TEST_F(TestLevelSetMultiObject, Weno5NeedsHalfWidth3)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(8.0f, Vec3f(0.0f), 1.0f, 2.0f);
    auto m = MLS::fromSignedGrids(in, 2);
    ASSERT_EQ(m->halfWidth(), 2);
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    using Evolver = lsmo_frames::FrameEvolver<MLS, Forces>;
    // the default HJWENO5 needs a band of 3 voxels: construction throws
    EXPECT_THROW(Evolver(*m, forces), ValueError);
    EXPECT_THROW(Evolver(*m, forces, math::HJWENO5_BIAS), ValueError);
    EXPECT_THROW(tools::LabelAwareStencil<MLS>(*m, math::HJWENO5_BIAS), ValueError);
    Evolver lsmo(*m, forces, math::FIRST_BIAS);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS);
    EXPECT_THROW(lsmo.setSpatialScheme(math::HJWENO5_BIAS), ValueError);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS); // unchanged by the throw
    EXPECT_GT(lsmo.step(1.0f), 0.0f);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, StencilWeno5Plane)
{
    // psi of label 1 on a tilted plane is linear: HJWENO5 one-sided
    // differences reproduce the gradient exactly on both sides (they read
    // +-3 voxels along each axis), as do the first-order ones.
    const Vec3d n = Vec3d(0.3, -0.5, 0.8).unitSafe();
    const double dx = 0.5;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3.0 * dx));
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    {
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        for (auto p = CoordBBox(Coord(-12), Coord(12)).begin(); p; ++p) {
            const double d = n.dot(xform->indexToWorld(*p)) + 0.1 * dx; // signed, label 1 < 0
            if (std::abs(d) >= 3.0 * dx) continue;
            pa.setValue(*p, float(std::abs(d)));
            la.setValue(*p, d < 0.0 ? 1 : 2);
        }
    }
    MLS m(phi, lab, 3);
    tools::LabelAwareStencil<MLS> s(m, math::HJWENO5_BIAS);
    EXPECT_EQ(s.spatialScheme(), math::HJWENO5_BIAS);
    s.moveTo(Coord(0), 1);
    float dneg[3], dpos[3], wneg[3], wpos[3], fneg[3], fpos[3];
    s.upwindDifferences(dneg, dpos);
    s.upwindDifferencesWeno5(wneg, wpos);
    s.upwindDifferencesFirstOrder(fneg, fpos);
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(wneg[a], n[a], 1e-4) << a;
        EXPECT_NEAR(wpos[a], n[a], 1e-4) << a;
        EXPECT_NEAR(fneg[a], n[a], 1e-4) << a;
        EXPECT_NEAR(fpos[a], n[a], 1e-4) << a;
        EXPECT_EQ(dneg[a], wneg[a]) << a; // the stencil's scheme
        EXPECT_EQ(dpos[a], wpos[a]) << a;
    }
    // first-order stencil: upwindDifferences is the first-order one
    tools::LabelAwareStencil<MLS> f(m);
    f.moveTo(Coord(0), 1);
    f.upwindDifferences(dneg, dpos);
    for (int a = 0; a < 3; ++a) {
        EXPECT_EQ(dneg[a], fneg[a]) << a;
        EXPECT_EQ(dpos[a], fpos[a]) << a;
    }
}

TEST_F(TestLevelSetMultiObject, Weno5ReducesAdvectionError)
{
    // The SingleObjectAdvection scenario (rigid rotation by 1.5 radians):
    // HJWENO5 / TVD-RK2 (the defaults) end closer to the exact sphere than
    // FIRST_BIAS / TVD-RK1, symmetric difference below 0.6x.
    Vec3SGrid::Ptr vel = makeRotation();
    const double angle = 0.05 * 30.0;
    FloatGrid::Ptr exact = tools::createLevelSetSphere<FloatGrid>(float(kEqRadius),
        Vec3f(float(kEqCenter[0] * std::cos(angle)), float(kEqCenter[0] * std::sin(angle)), 0.0f),
        1.0f);
    double err[2];
    for (int k = 0; k < 2; ++k) {
        auto m = makeSphere(kEqRadius, kEqCenter);
        Forces forces;
        forces.setAdvection(vel, 1.0f);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        if (k == 0) pinFirstOrder(lsmo);
        const auto start = std::chrono::steady_clock::now();
        const size_t steps = lsmo.advect(0.0, 30.0);
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const ZeroSetDiff d = compareZeroSets(*m->extractSignedDistance(1), *exact);
        err[k] = d.symDiff;
        std::cout << "Weno5ReducesAdvectionError: " << (k == 0 ? "FIRST_BIAS/TVD_RK1" :
            "HJWENO5_BIAS/TVD_RK2") << ": " << steps << " steps, " << 1e3 * secs / double(steps)
                  << " ms/step, symmetric difference vs exact " << d.symDiff << " Hausdorff "
                  << d.hausdorff << " dx, Euler fallbacks " << lsmo.eulerFallbacks()
                  << std::endl;
    }
    EXPECT_LT(err[1], 0.6 * err[0]);
}

TEST_F(TestLevelSetMultiObject, SplittingFixedAndPassive)
{
    // Constant pressure field: dpsi_1/dt = -|grad psi_1|, unit growth.
    FloatGrid::Ptr pressure = FloatGrid::create(1.0f);
    // Curvature weight 0.5: the parabolic limit dx^2/(6*0.5) = 1/3 is below
    // the hyperbolic dt (about 0.5*cfl*dx). An unsplit step is at most
    // stableDt = 1/3, so a step longer than 1/3 was split.
    const float weight = 0.5f;
    const float pStable = 1.0f / (6.0f * weight);
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    auto contactCount = [&](const MLS& m) {
        Index64 contact = 0;
        for (auto it = m.labels()->tree().cbeginValueOn(); it; ++it) {
            if (*it != 1) continue;
            for (int f = 0; f < 6; ++f) {
                contact += m.labels()->tree().getValue(it.getCoord() + offs[f]) == 2;
            }
        }
        return contact;
    };

    {
        // FixedLabels with curvature + pressure, split: the fixed label's
        // voxels are bit-identical and the moving label reaches it; the
        // first step and the steps around first contact are split.
        auto m = makeTwoSpheres(1.0);
        std::map<Coord, float> before;
        for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
            if (*it == 2) before[it.getCoord()] = m->phi()->tree().getValue(it.getCoord());
        }
        ASSERT_FALSE(before.empty());
        Forces forces;
        forces.setPressure(pressure, 1.0f);
        forces.setCurvatureWeight(weight);
        forces.setFixedLabels({2});
        ASSERT_EQ(forces.parabolicStableDt(1.0f), pStable);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        ASSERT_TRUE(lsmo.splitting());
        std::vector<float> dts;
        int firstContact = -1;
        for (int n = 0; n < 30; ++n) {
            const float dt = lsmo.step(0.75f);
            ASSERT_GT(dt, 0.0f);
            dts.push_back(dt);
            std::string msg;
            ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
            if (firstContact < 0 && contactCount(*m) > 0) firstContact = n;
        }
        std::cout << "SplittingFixedAndPassive: fixed: first contact after step " << firstContact
                  << ", dt:";
        for (float dt : dts) std::cout << " " << dt;
        std::cout << std::endl;
        ASSERT_GE(firstContact, 1);
        EXPECT_GT(dts[0], pStable) << "the first step is not split";
        for (int n = firstContact - 1; n <= std::min(firstContact + 1, 29); ++n) {
            EXPECT_GT(dts[n], pStable) << "step " << n << " (around first contact) is not split";
        }
        Index64 changed = 0;
        for (const auto& kv : before) {
            const float phi = m->phi()->tree().getValue(kv.first);
            if (!m->labels()->tree().isValueOn(kv.first)
                || m->labels()->tree().getValue(kv.first) != 2
                || std::memcmp(&phi, &kv.second, sizeof(float)) != 0) ++changed;
        }
        EXPECT_EQ(changed, Index64(0));
        EXPECT_GT(contactCount(*m), Index64(0));
        EXPECT_EQ(countOrphans(*m), Index64(0));
    }
    {
        // PressureGrowth with curvature, split, against passive label 0:
        // dr/dt = 1 - w/r; the radius at t = 6 is within the PressureGrowth
        // gate (0.5*dx) of that ODE's solution.
        auto m = makeSphere(8.0);
        const double dx = m->voxelSize();
        Forces forces;
        forces.setPressure(pressure, 1.0f);
        forces.setCurvatureWeight(weight);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        ASSERT_TRUE(lsmo.splitting());
        const double tEnd = 6.0;
        std::vector<float> dts;
        double t = 0.0;
        while (tEnd - t > 1e-6) {
            const float dt = lsmo.step(float(tEnd - t));
            ASSERT_GT(dt, 0.0f);
            dts.push_back(dt);
            t += dt;
        }
        size_t split = 0;
        for (float dt : dts) split += dt > pStable;
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        double exact = 8.0; // dr/dt = 1 - w/r, RK4
        const int kOdeSteps = 6000;
        const double h = tEnd / kOdeSteps;
        auto f = [&](double r) { return 1.0 - double(weight) / r; };
        for (int i = 0; i < kOdeSteps; ++i) {
            const double k1 = f(exact), k2 = f(exact + 0.5 * h * k1),
                k3 = f(exact + 0.5 * h * k2), k4 = f(exact + h * k3);
            exact += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }
        const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
        std::cout << "SplittingFixedAndPassive: growth: steps " << dts.size() << " (split "
                  << split << ") mean radius " << r << " (expected " << exact << ")"
                  << std::endl;
        EXPECT_GT(dts[0], pStable) << "the first step is not split";
        EXPECT_GE(split + 1, dts.size()); // all but possibly a short last step
        EXPECT_NEAR(r, exact, 0.5 * dx);
    }
}


////////////////////////////////////////
// Eight objects in a vortex.

namespace {

Index64
countNaN(const FloatGrid& g)
{
    Index64 n = 0;
    for (auto it = g.tree().cbeginValueAll(); it; ++it) n += std::isnan(*it) ? 1 : 0;
    return n;
}

// Voxels with psi < 0 of a level set grid (active and inactive, tiles by
// voxel count).
Index64
insideVoxelCount(const FloatGrid& g)
{
    Index64 n = 0;
    for (auto it = g.tree().cbeginValueAll(); it; ++it) n += *it < 0.0f ? it.getVoxelCount() : 0;
    return n;
}

} // anonymous namespace

namespace {

// The VortexEightObjects scenario and gates; @a motion as for
// runPressureGrowth.
void
runVortexEightObjects(double motion)
{
    // Scene (lsmo_test::Vortex):
    // - eight shapes, labels 1-8, every length times Vortex::kShapeScale =
    //   1.4 (sphere r = 14; cube side 16.8; tetrahedron, octahedron,
    //   dodecahedron, icosahedron with circumradius 14; capsule along z with
    //   radius 7 and axis endpoints +-7; closed ring = 8-segment polygon of
    //   radius 7 in the xy plane with tube radius 4.2). No rotations.
    // - equally spaced on a circle near the periphery of the vortex: label l
    //   centered at angle (l-1)*45 degrees, r = 34, z = 0; dx = 1,
    //   halfWidth 3.
    // - Vortex field (see Vortex): a volume-preserving inward spiral towards
    //   the center of the [-80, 80]^3 velocity grid: inflow 0.4 (vanishing
    //   inside r ~ 8), solid-body swirl 0.012 rad per unit time, axial
    //   outflow cancelling the convergence; advection weight 1, curvature
    //   weight 0.003.
    // - Why these values: an axisymmetric inflow keeps the angular gaps of
    //   objects on one circle, so neighbors touch only once their gaps
    //   (proportional to r) close below a voxel. r = 34 leaves five
    //   neighbor pairs 1.1 to 1.7 voxels apart (the ring's polygon radius
    //   was reduced from 7 to 5 so that it does not touch the sphere at the
    //   start), so contacts form in the outer part of the spiral, before the
    //   objects are strongly stretched near the axis. Rigid rotation (rather
    //   than a Rankine swirl) keeps neighbors from sliding apart once in
    //   contact. A stronger inflow brings the objects deep into the core,
    //   where the stretching costs every object a third or more of its
    //   volume; a weaker one gives contact late. With the default schemes
    //   contact (>= 4 pairs) comes at t = 34 and the smallest label keeps
    //   ~950 voxels at t = 60 (tetrahedron, label 3).
    // - Advect in chunks of 2 until >= 4 contact pairs (capped at t = 70),
    //   then on to t = 60 (Vortex::kEndTime); contacts must stay >= 4 at
    //   every chunk after first contact (>= 3 at full-voxel motion, whose
    //   coarser steps erode the one-voxel contacts of sharp corners).
    // Volume gate (two-sided): with refLoss the loss of the same
    // shape advected alone by LevelSetAdvection with the evolver's schemes
    // (the defaults, HJWENO5 / TVD-RK2) in the
    // same field over the same time, each label's voxel loss lies in
    // [1.1*min(refLoss, 0) - slack, 1.1*max(refLoss, 0) + slack], slack 2%
    // (5% at full-voxel motion, where the ring squeezed against the sphere
    // loses ~6 points more than alone). The exact flow preserves volume;
    // both runs lose volume numerically, mostly where objects stretch.
    // (Shapes, placement, field and schedule: lsmo_test::Vortex.)
    // Contact pairs required after first contact, and the volume gate's
    // slack (fraction), at half-voxel and at full-voxel motion.
    constexpr size_t kMinPairsAfterContact = 4, kMinPairsAfterContactFullVoxel = 3;
    constexpr double kVolumeSlack = 0.02, kVolumeSlackFullVoxel = 0.05;
    const bool fullVoxel = motion > tools::lsmo_internal::kHalfVoxelMotion;
    const size_t minPairsAfter = fullVoxel ? kMinPairsAfterContactFullVoxel
                                           : kMinPairsAfterContact;
    const double volumeSlack = fullVoxel ? kVolumeSlackFullVoxel : kVolumeSlack;
    const Vortex vortex;
    const std::map<int32_t, FloatGrid::ConstPtr> in = vortex.inputs();
    auto m = MLS::fromSignedGrids(in, 3);
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
    const std::map<int32_t, Index64> vol0 = labelVoxelCounts(*m);
    ASSERT_TRUE(contactPairs(*m).empty());

    Vec3SGrid::Ptr vel = vortex.velocity();
    size_t steps = 0;
    double tContact = -1.0;
    std::set<std::pair<int32_t, int32_t>> pairs;
    // failures of this run only (FullVoxelMotion runs several scenarios)
    const int failures0 = failureCount();
    const auto start = std::chrono::steady_clock::now();
    math::BiasedGradientScheme spatial = math::UNKNOWN_BIAS;
    math::TemporalIntegrationScheme temporal = math::UNKNOWN_TIS;
    double usedMotion = 0.0;
    // Keep going to the end time; contacts must persist after first contact.
    const double t = vortex.run(*m, vel, steps, tContact,
        [&](double tc, size_t n, const std::set<std::pair<int32_t, int32_t>>& p, int after) {
            std::string msg;
            if (!m->validate(&msg)) { ADD_FAILURE() << "t " << tc << ": " << msg; return false; }
            Index64 overlaps = 0, badCover = 0;
            partitionErrors(*m, overlaps, badCover);
            EXPECT_EQ(overlaps, Index64(0)) << "t " << tc;
            EXPECT_EQ(badCover, Index64(0)) << "t " << tc;
            EXPECT_EQ(countNaN(*m->phi()), Index64(0)) << "t " << tc;
            // phi = 0 belongs to no label: the evolver never writes it
            Index64 zeros = 0;
            for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) zeros += *it == 0.0f;
            EXPECT_EQ(zeros, Index64(0)) << "t " << tc;
            pairs = p;
            std::cout << "Vortex: t " << tc << " steps " << n << " contacts " << p.size();
            for (auto& lv : labelVoxelCounts(*m)) std::cout << " " << lv.first << ":" << lv.second;
            std::cout << std::endl;
            if (after >= 1) { EXPECT_GE(p.size(), minPairsAfter) << "t " << tc; }
            return failureCount() == failures0;
        },
        [&](Vortex::Evolver& e) {
            applyMotion(e, motion);
            usedMotion = e.maxInterfaceMotion();
            spatial = e.spatialScheme();
            temporal = e.temporalScheme();
        });
    ASSERT_EQ(failureCount(), failures0);
    std::cout << "Vortex: >= 4 contact pairs first at t " << tContact << std::endl;
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Vortex: motion " << usedMotion << ", t " << t
              << ", " << steps << " steps, " << secs << " s ("
              << 1e3 * secs / double(std::max(steps, size_t(1))) << " ms/step), contact pairs";
    for (const auto& pr : pairs) std::cout << " " << pr.first << "-" << pr.second;
    std::cout << std::endl;
    EXPECT_GE(pairs.size(), size_t(4));

    const std::map<int32_t, Index64> vol1 = labelVoxelCounts(*m);
    for (int32_t l = 1; l <= 8; ++l) {
        // reference: the same shape alone, LevelSetAdvection with the
        // evolver's schemes to the same t
        FloatGrid::Ptr ref = in.at(l)->deepCopy();
        const double r0 = double(insideVoxelCount(*ref));
        referenceAdvect(*ref, *vel, 0.0, t, spatial, temporal);
        const double refLoss = 1.0 - double(insideVoxelCount(*ref)) / r0;
        const double v0 = double(vol0.at(l));
        const double v1 = vol1.count(l) ? double(vol1.at(l)) : 0.0;
        const double loss = 1.0 - v1 / v0;
        std::cout << "Vortex: label " << l << " voxels " << v0 << " -> " << v1 << " (loss "
                  << 100.0 * loss << "%), reference loss " << 100.0 * refLoss << "%" << std::endl;
        EXPECT_GE(v1, 1.0) << "label " << l;
        // two-sided (a reference gain bounds MOGAC's gain too)
        EXPECT_LE(loss, 1.1 * std::max(refLoss, 0.0) + volumeSlack) << "label " << l;
        EXPECT_GE(loss, 1.1 * std::min(refLoss, 0.0) - volumeSlack) << "label " << l;
    }
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, VortexEightObjects)
{
    runVortexEightObjects(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, Weno5ReducesVortexLoss)
{
    // The VortexEightObjects run with the defaults (HJWENO5 / TVD-RK2) and
    // with FIRST_BIAS / TVD-RK1. The field preserves volume, so the error of
    // label l is |loss_l| (loss 1 - V(t)/V(0) in voxels). The worst error of
    // the default run is below 0.6x the first-order run's. (The losses of
    // the same shape advected alone by LevelSetAdvection with each scheme
    // are printed only.)
    const Vortex vortex;
    Vec3SGrid::Ptr vel = vortex.velocity();
    const std::map<int32_t, FloatGrid::ConstPtr> in = vortex.inputs();
    struct Run { double t = 0.0; std::map<int32_t, double> loss; };
    Run runs[2];
    for (int k = 0; k < 2; ++k) {
        auto m = MLS::fromSignedGrids(in, 3);
        const std::map<int32_t, Index64> vol0 = labelVoxelCounts(*m);
        size_t steps = 0;
        double tContact = -1.0;
        const auto start = std::chrono::steady_clock::now();
        runs[k].t = vortex.run(*m, vel, steps, tContact,
            [&](double, size_t, const std::set<std::pair<int32_t, int32_t>>&, int) { return true; },
            [k](Vortex::Evolver& e) { if (k == 0) pinFirstOrder(e); });
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const std::map<int32_t, Index64> vol1 = labelVoxelCounts(*m);
        for (int32_t l = 1; l <= 8; ++l) {
            const double v1 = vol1.count(l) ? double(vol1.at(l)) : 0.0;
            runs[k].loss[l] = 1.0 - v1 / double(vol0.at(l));
        }
        std::cout << "Weno5ReducesVortexLoss: " << (k == 0 ? "FIRST_BIAS/TVD_RK1" :
            "HJWENO5_BIAS/TVD_RK2") << ": t " << runs[k].t << ", " << steps << " steps, "
                  << 1e3 * secs / double(steps) << " ms/step, first contact " << tContact
                  << std::endl;
    }
    ASSERT_EQ(runs[0].t, runs[1].t);
    // single-shape references: [0] FIRST_BIAS / TVD_RK1, [1] HJWENO5 / TVD_RK2
    std::map<int32_t, double> refLoss[2];
    for (int k = 0; k < 2; ++k) {
        for (int32_t l = 1; l <= 8; ++l) {
            FloatGrid::Ptr g = in.at(l)->deepCopy();
            const double r0 = double(insideVoxelCount(*g));
            referenceAdvect(*g, *vel, 0.0, runs[k].t, k == 0 ? math::FIRST_BIAS : math::HJWENO5_BIAS,
                k == 0 ? math::TVD_RK1 : math::TVD_RK2);
            refLoss[k][l] = 1.0 - double(insideVoxelCount(*g)) / r0;
        }
    }
    double worst[2] = {0.0, 0.0}, worstOwn[2] = {0.0, 0.0};
    std::cout << "Weno5ReducesVortexLoss: label | loss FIRST/RK1, HJWENO5/RK2 | reference loss"
                 " FIRST/RK1, HJWENO5/RK2 | error (|loss|) FIRST, HJWENO5"
                 " | error vs own-scheme reference FIRST, HJWENO5 (percent)" << std::endl;
    for (int32_t l = 1; l <= 8; ++l) {
        double err[2], own[2];
        for (int k = 0; k < 2; ++k) {
            err[k] = std::abs(runs[k].loss[l]);
            own[k] = std::abs(runs[k].loss[l] - refLoss[k][l]);
            worst[k] = std::max(worst[k], err[k]);
            worstOwn[k] = std::max(worstOwn[k], own[k]);
        }
        std::cout << "Weno5ReducesVortexLoss: " << l << " | " << 100.0 * runs[0].loss[l] << ", "
                  << 100.0 * runs[1].loss[l] << " | " << 100.0 * refLoss[0][l] << ", "
                  << 100.0 * refLoss[1][l] << " | " << 100.0 * err[0] << ", " << 100.0 * err[1]
                  << " | " << 100.0 * own[0] << ", " << 100.0 * own[1] << std::endl;
    }
    std::cout << "Weno5ReducesVortexLoss: worst error (|loss|): FIRST "
              << 100.0 * worst[0] << ", HJWENO5 " << 100.0 * worst[1] << " ("
              << worst[1] / worst[0] << "x); vs own-scheme reference: FIRST "
              << 100.0 * worstOwn[0] << ", HJWENO5 " << 100.0 * worstOwn[1] << std::endl;
    EXPECT_LT(worst[1], 0.6 * worst[0]);
}


////////////////////////////////////////
// Volume constraints

namespace {

// (LabelSpeed: per-label constant dpsi/dt, defined above.)
bool allPhiFinite(const MLS& m)
{
    for (auto it = m.phi()->tree().cbeginValueAll(); it; ++it) {
        if (!std::isfinite(*it)) return false;
    }
    return true;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, VolumeConstraintApi)
{
    using Evolver = lsmo_frames::FrameEvolver<MLS, LabelSpeed>;
    // labelVolume: a sphere of radius 8 (dx 1 and 0.5) within 1% of 4/3 pi r^3;
    // an absent label has volume 0
    const double r = 8.0;
    const double exact = 4.0 / 3.0 * math::pi<double>() * r * r * r;
    for (const double dx : {1.0, 0.5}) {
        auto m = makeSphere(r, Vec3d(0.0), dx);
        LabelSpeed speed;
        Evolver e(*m, speed);
        const double v = e.labelVolume(1);
        std::cout << "VolumeConstraintApi: dx " << dx << " sphere volume " << v << " (exact "
                  << exact << ")" << std::endl;
        EXPECT_NEAR(v, exact, 0.01 * exact) << "dx " << dx;
        EXPECT_EQ(e.labelVolume(7), 0.0);
    }

    auto m = makeSphere(r);
    LabelSpeed speed;
    speed.fixed = {2};
    Evolver e(*m, speed);
    EXPECT_DOUBLE_EQ(e.volumeRelaxation(), 0.5);
    EXPECT_THROW(e.setVolumeConstraint(0, 1.0), ValueError);  // passive label 0
    EXPECT_THROW(e.preserveInitialVolume(0), ValueError);
    EXPECT_THROW(e.setVolumeConstraint(2, 1.0), ValueError);  // fixed
    EXPECT_THROW(e.setVolumeConstraint(1, -1.0), ValueError); // negative target
    EXPECT_THROW(e.setVolumeConstraint(1, std::numeric_limits<double>::quiet_NaN()), ValueError);
    EXPECT_THROW(e.setVolumeRelaxation(0.0), ValueError);
    EXPECT_THROW(e.setVolumeRelaxation(1.5), ValueError);
    auto still = [](double) { return math::Mat4d::identity(); };
    e.setKinematicLabel(3, tools::createLevelSetSphere<FloatGrid>(2.0f, Vec3f(20.0f), 1.0f), still);
    EXPECT_THROW(e.setVolumeConstraint(3, 1.0), ValueError);  // kinematic
    EXPECT_THROW(e.preserveInitialVolume(3), ValueError);
    e.clearKinematicLabel(3);
    e.preserveInitialVolume(1);
    EXPECT_THROW(e.setKinematicLabel(1, tools::createLevelSetSphere<FloatGrid>(2.0f,
        Vec3f(20.0f), 1.0f), still), ValueError); // constrained labels cannot become kinematic
    EXPECT_EQ(e.volumeConstraints().size(), size_t(1));
    EXPECT_DOUBLE_EQ(e.volumeConstraints().at(1), e.labelVolume(1));

    // Growth to a target with zero speed: the constraint alone moves the
    // interface, to within 1% of the target of a radius-9 sphere's volume.
    const double target = 4.0 / 3.0 * math::pi<double>() * 729.0;
    e.setVolumeConstraint(1, target);
    for (int n = 0; n < 20; ++n) {
        const float dt = e.step(1.0f);
        ASSERT_TRUE(std::isfinite(dt) && dt > 0.0f);
    }
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const double v = e.labelVolume(1);
    std::cout << "VolumeConstraintApi: grown to " << v << " (target " << target << ")" << std::endl;
    EXPECT_NEAR(v, target, 0.01 * target);
    EXPECT_NEAR(meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0)), 9.0, 0.25);

    // Cleared: zero speed leaves the volume where it is.
    e.clearVolumeConstraints();
    EXPECT_TRUE(e.volumeConstraints().empty());
    for (int n = 0; n < 5; ++n) e.step(1.0f);
    EXPECT_NEAR(e.labelVolume(1), v, 1e-3 * v);
}

TEST_F(TestLevelSetMultiObject, VolumeConstraintCancelsDrift)
{
    // A sphere of radius 8 under a uniform shrinking speed (dpsi/dt = 0.5)
    // for t = 10: unconstrained it loses most of its volume; constrained to
    // its initial volume the mean-speed term cancels the drift, so it ends
    // within 2% (a proportional correction alone would lag by about one
    // step's drift divided by the relaxation).
    double change[2] = {0.0, 0.0};
    for (int k = 0; k < 2; ++k) {
        auto m = makeSphere(8.0);
        LabelSpeed speed;
        speed.speeds[1] = 0.5f;
        lsmo_frames::FrameEvolver<MLS, LabelSpeed> e(*m, speed);
        const double v0 = e.labelVolume(1);
        if (k == 1) e.preserveInitialVolume(1);
        e.advect(0.0, 10.0);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        change[k] = e.labelVolume(1) / v0 - 1.0;
        std::cout << "VolumeConstraintCancelsDrift: " << (k ? "constrained" : "unconstrained")
                  << " volume change " << 100.0 * change[k] << "%" << std::endl;
    }
    EXPECT_LT(change[0], -0.5);
    EXPECT_LE(std::abs(change[1]), 0.02);
}

TEST_F(TestLevelSetMultiObject, VolumeConstraintVanishingLabel)
{
    // Three separate spheres: label 1 (radius 4) vanishes under a strong
    // inward speed (dpsi/dt = 2) while label 2 (radius 6, zero speed) keeps
    // its initial volume; label 3 (radius 4, zero speed) is driven to a
    // target volume of 0 by its own constraint and vanishes too. Every step
    // keeps phi and the volumes finite and validate() true; the vanished
    // labels then have volume 0 (their constraint is skipped).
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(-14.0f, 0.0f, 0.0f), 1.0f);
    in[2] = tools::createLevelSetSphere<FloatGrid>(6.0f, Vec3f(0.0f), 1.0f);
    in[3] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(14.0f, 0.0f, 0.0f), 1.0f);
    auto m = MLS::fromSignedGrids(in);
    LabelSpeed speed;
    speed.speeds[1] = 2.0f;
    lsmo_frames::FrameEvolver<MLS, LabelSpeed> e(*m, speed);
    e.preserveInitialVolume(2);
    e.setVolumeConstraint(3, 0.0);
    const double v2 = e.labelVolume(2);
    int gone1 = -1, gone3 = -1;
    for (int n = 0; n < 60; ++n) {
        const float dt = e.step(1.0f);
        ASSERT_TRUE(std::isfinite(dt) && dt > 0.0f) << "step " << n;
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
        ASSERT_TRUE(allPhiFinite(*m)) << "step " << n;
        for (int32_t l = 1; l <= 3; ++l) {
            ASSERT_TRUE(std::isfinite(e.labelVolume(l))) << "step " << n << " label " << l;
        }
        const auto ls = m->labelSet();
        if (gone1 < 0 && !std::count(ls.begin(), ls.end(), 1)) gone1 = n;
        if (gone3 < 0 && !std::count(ls.begin(), ls.end(), 3)) gone3 = n;
    }
    std::cout << "VolumeConstraintVanishingLabel: label 1 gone at step " << gone1
              << ", label 3 at step " << gone3 << ", label 2 volume " << v2 << " -> "
              << e.labelVolume(2) << std::endl;
    EXPECT_GE(gone1, 0) << "label 1 did not vanish";
    EXPECT_GE(gone3, 0) << "label 3 did not vanish";
    EXPECT_EQ(e.labelVolume(1), 0.0);
    EXPECT_EQ(e.labelVolume(3), 0.0);
    EXPECT_NEAR(e.labelVolume(2), v2, 0.02 * v2);
    EXPECT_EQ(countOrphans(*m), Index64(0));
}

TEST_F(TestLevelSetMultiObject, VolumeConstraintVortex)
{
    // The VortexEightObjects run (default schemes, same schedule) three
    // times: unconstrained (baseline), with every label constrained to its
    // initial volume (preserveInitialVolume, default relaxation 0.5), and
    // mixed, with labels 1, 2 and 8 constrained (their contacts, 1-2 and
    // 1-8, are among themselves) and 3-7 not (contacts 3-4, 4-5 and 5-6;
    // neither group touches the other). Volumes are the evolver's
    // labelVolume().
    // Gates: constrained labels end within 2% of their initial volume
    // (all-constrained and mixed runs); >= 4 contact pairs at every chunk
    // after first contact and validate() at every chunk; in the mixed run
    // the unconstrained labels (which touch only each other) end within 1
    // percentage point of their baseline volume change.
    const Vortex vortex;
    const std::map<int32_t, FloatGrid::ConstPtr> in = vortex.inputs();
    Vec3SGrid::Ptr vel = vortex.velocity();
    struct Run
    {
        std::map<int32_t, double> v0, v1, worst;
        std::map<int32_t, Index64> n0, n1;
        size_t steps = 0;
        double t = 0.0, tContact = -1.0, secs = 0.0;
        size_t minPairsAfter = 0;
    };
    auto run = [&](const std::set<int32_t>& constrained, const char* name) {
        Run res;
        auto m = MLS::fromSignedGrids(in, 3);
        res.n0 = labelVoxelCounts(*m);
        Vortex::Evolver* ev = nullptr;
        res.minPairsAfter = std::numeric_limits<size_t>::max();
        const auto start = std::chrono::steady_clock::now();
        res.t = vortex.run(*m, vel, res.steps, res.tContact,
            [&](double tc, size_t, const std::set<std::pair<int32_t, int32_t>>& p, int after) {
                std::string msg;
                if (!m->validate(&msg)) {
                    ADD_FAILURE() << name << " t " << tc << ": " << msg;
                    return false;
                }
                if (!allPhiFinite(*m)) {
                    ADD_FAILURE() << name << " t " << tc << ": NaN or inf phi";
                    return false;
                }
                if (after >= 1) res.minPairsAfter = std::min(res.minPairsAfter, p.size());
                for (int32_t l = 1; l <= 8; ++l) {
                    const double v = ev->labelVolume(l);
                    res.v1[l] = v;
                    res.worst[l] = std::max(res.worst[l], std::abs(v / res.v0[l] - 1.0));
                }
                return true;
            },
            [&](Vortex::Evolver& e) {
                ev = &e;
                for (int32_t l = 1; l <= 8; ++l) res.v0[l] = e.labelVolume(l);
                for (const int32_t l : constrained) e.preserveInitialVolume(l);
            });
        res.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        res.n1 = labelVoxelCounts(*m);
        std::cout << "VolumeConstraintVortex: " << name << ": t " << res.t << ", " << res.steps
                  << " steps, " << res.secs << " s (" << 1e3 * res.secs / double(res.steps)
                  << " ms/step), first contact " << res.tContact << ", min pairs after contact "
                  << res.minPairsAfter << std::endl;
        for (int32_t l = 1; l <= 8; ++l) {
            std::cout << "VolumeConstraintVortex: " << name << ": label " << l
                      << (constrained.count(l) ? " (constrained)" : "") << " volume "
                      << res.v0[l] << " -> " << res.v1[l] << " ("
                      << 100.0 * (res.v1[l] / res.v0[l] - 1.0) << "%, worst over chunks "
                      << 100.0 * res.worst[l] << "%), voxels " << res.n0[l] << " -> "
                      << (res.n1.count(l) ? res.n1[l] : 0) << std::endl;
        }
        return res;
    };
    const Run base = run({}, "unconstrained");
    ASSERT_FALSE(::testing::Test::HasFailure());
    const Run all = run({1, 2, 3, 4, 5, 6, 7, 8}, "all constrained");
    ASSERT_FALSE(::testing::Test::HasFailure());
    const std::set<int32_t> mixedSet{1, 2, 8};
    const Run mixed = run(mixedSet, "mixed");
    ASSERT_FALSE(::testing::Test::HasFailure());

    EXPECT_GE(all.tContact, 0.0);
    EXPECT_GE(all.minPairsAfter, size_t(4));
    EXPECT_GE(mixed.minPairsAfter, size_t(4));
    EXPECT_EQ(all.t, Vortex::kEndTime);
    for (int32_t l = 1; l <= 8; ++l) {
        EXPECT_LE(std::abs(all.v1.at(l) / all.v0.at(l) - 1.0), 0.02) << "label " << l;
        const double change = mixed.v1.at(l) / mixed.v0.at(l) - 1.0;
        if (mixedSet.count(l)) {
            EXPECT_LE(std::abs(change), 0.02) << "mixed, label " << l;
        } else {
            const double baseChange = base.v1.at(l) / base.v0.at(l) - 1.0;
            EXPECT_LE(std::abs(change - baseChange), 0.01) << "mixed, unconstrained label " << l;
        }
    }
}


////////////////////////////////////////
// Topology-preserving evolution

namespace {

using Conn = tools::MultiObjectConnectivity;
const Conn kAllConn[4] = {Conn::FG26_BG6, Conn::FG6_BG26, Conn::FG18_BG6, Conn::FG6_BG18};

uint32_t
nbBit(int a, int b, int c)
{
    return uint32_t(1) << ((a + 1) + 3 * (b + 1) + 9 * (c + 1));
}

// Reference topological numbers, written straight from the Bertrand-Malandain
// definitions on coordinates (independent of the bitmask implementation):
// N^1_n(x,X) = N*_n(x) & X, N^k_n = union over y in N^{k-1} of
// N_n(y) & N*_26(x) & X; G_6 = N^2_6, G_6+ = N^3_6, G_18 = N^2_18,
// G_26 = N^1_26; T_n = number of n-components of G_n.
int
refTopoNumber(uint32_t inSet, int n, int k)
{
    auto d2 = [](const Coord& a, const Coord& b) { return (a - b).asVec3i().lengthSqr(); };
    const int bound = n == 6 ? 1 : (n == 18 ? 2 : 3);
    std::vector<Coord> X;
    for (int c = -1; c <= 1; ++c)
        for (int b = -1; b <= 1; ++b)
            for (int a = -1; a <= 1; ++a)
                if ((a || b || c) && (inSet & nbBit(a, b, c))) X.push_back(Coord(a, b, c));
    std::set<Coord> g;
    for (const Coord& p : X) if (d2(p, Coord(0)) <= bound) g.insert(p);
    for (int level = 2; level <= k; ++level) {
        std::set<Coord> next = g;
        for (const Coord& y : g)
            for (const Coord& p : X) if (d2(p, y) <= bound) next.insert(p);
        g = next;
    }
    // n-components of g
    std::vector<Coord> pts(g.begin(), g.end());
    std::vector<int> comp(pts.size(), -1);
    int count = 0;
    for (size_t s = 0; s < pts.size(); ++s) {
        if (comp[s] >= 0) continue;
        std::vector<size_t> stack{s};
        comp[s] = count;
        while (!stack.empty()) {
            const size_t c = stack.back();
            stack.pop_back();
            for (size_t q = 0; q < pts.size(); ++q) {
                if (comp[q] < 0 && d2(pts[q], pts[c]) <= bound) { comp[q] = count; stack.push_back(q); }
            }
        }
        ++count;
    }
    return count;
}

bool
refSimple(uint32_t mask, Conn conn)
{
    const uint32_t all = (uint32_t(1) << 27) - 1, center = nbBit(0, 0, 0);
    const uint32_t fg = mask & all & ~center, bg = ~mask & all & ~center;
    switch (conn) {
    case Conn::FG26_BG6: return refTopoNumber(fg, 26, 1) == 1 && refTopoNumber(bg, 6, 2) == 1;
    case Conn::FG6_BG26: return refTopoNumber(fg, 6, 2) == 1 && refTopoNumber(bg, 26, 1) == 1;
    case Conn::FG18_BG6: return refTopoNumber(fg, 18, 2) == 1 && refTopoNumber(bg, 6, 3) == 1;
    case Conn::FG6_BG18: return refTopoNumber(fg, 6, 3) == 1 && refTopoNumber(bg, 18, 2) == 1;
    }
    return false;
}

Int32Grid::Ptr
boxLabels(const std::vector<std::pair<CoordBBox, int32_t>>& boxes)
{
    Int32Grid::Ptr g = Int32Grid::create(0);
    auto acc = g->getAccessor();
    for (const auto& bl : boxes) {
        for (auto p = bl.first.begin(); p; ++p) acc.setValue(*p, bl.second);
    }
    return g;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, SimplePointTable)
{
    using tools::lsmo_internal::isSimplePoint;
    const uint32_t c = nbBit(0, 0, 0);
    uint32_t ring = c, plug = 0, face = 0, full = 0;
    for (int k = -1; k <= 1; ++k)
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                full |= nbBit(i, j, k);
                if (k == 0) ring |= nbBit(i, j, k);
                if (k <= 0) face |= nbBit(i, j, k);
                if (!(i == 0 && j == 0 && k != 0)) plug |= nbBit(i, j, k);
            }
    for (Conn conn : kAllConn) {
        SCOPED_TRACE(int(conn));
        EXPECT_FALSE(isSimplePoint(c, conn));                                     // isolated voxel
        EXPECT_TRUE(isSimplePoint(c | nbBit(1, 0, 0), conn));                     // end of a line
        EXPECT_FALSE(isSimplePoint(c | nbBit(-1, 0, 0) | nbBit(1, 0, 0), conn));  // middle of a line
        EXPECT_FALSE(isSimplePoint(ring, conn));  // completes a ring around a tunnel
        EXPECT_FALSE(isSimplePoint(plug, conn));  // plugs the last hole of a cavity
        EXPECT_FALSE(isSimplePoint(full, conn));  // interior voxel (removal makes a cavity)
        EXPECT_TRUE(isSimplePoint(face, conn));   // face of a solid block
        // the center bit is ignored
        EXPECT_EQ(isSimplePoint(face, conn), isSimplePoint(face & ~c, conn));
    }
    // x joins two opposite corner voxels that are not adjacent to each other
    const uint32_t corner = c | nbBit(-1, -1, -1) | nbBit(1, 1, 1);
    EXPECT_FALSE(isSimplePoint(corner, Conn::FG26_BG6)); // two 26-components in N*
    // 10^5 random neighborhoods against the reference definitions, at
    // several densities
    std::mt19937 gen(20261109u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    Index64 mismatches = 0, simple = 0;
    for (int n = 0; n < 100000; ++n) {
        const double density = 0.1 + 0.8 * uni(gen);
        uint32_t mask = 0;
        for (int b = 0; b < 27; ++b) if (uni(gen) < density) mask |= uint32_t(1) << b;
        const Conn conn = kAllConn[n % 4];
        const bool got = isSimplePoint(mask, conn);
        simple += got;
        if (got != refSimple(mask, conn)) ++mismatches;
    }
    std::cout << "SimplePointTable: " << simple << " of 100000 random masks simple" << std::endl;
    EXPECT_EQ(mismatches, Index64(0));
    EXPECT_GT(simple, Index64(1000)); // the random set exercises both outcomes
}

TEST_F(TestLevelSetMultiObject, DigitalTopologyHelper)
{
    // Region numbers under each rule, checked against the rule's mesh.
    const CoordBBox dom(Coord(-2), Coord(13));
    auto expectTopo = [&](const char* name, const Int32Grid& g, Conn conn, int comps, int cav,
                          int euler) {
        auto m = MLS::fromLabelGrid(g);
        const lsmo_test::DigitalTopology expected{comps, cav, euler};
        EXPECT_TOPOLOGY_CONSISTENT(lsmo_test::verifyLabelTopology(*m, 1, dom, conn,
            std::string(name) + "/" + lsmo_test::connectivityName(conn), &expected));
    };
    auto solid = boxLabels({{CoordBBox(Coord(0), Coord(7)), 1}});
    auto hollow = boxLabels({{CoordBBox(Coord(0), Coord(7)), 1}, {CoordBBox(Coord(2), Coord(5)), 0}});
    auto torus = boxLabels({{CoordBBox(Coord(0, 0, 0), Coord(9, 9, 3)), 1},
        {CoordBBox(Coord(3, 3, 0), Coord(6, 6, 3)), 0}});
    auto corner = boxLabels({{CoordBBox(Coord(0), Coord(2)), 1}, {CoordBBox(Coord(3), Coord(5)), 1}});
    auto edge = boxLabels({{CoordBBox(Coord(0), Coord(2)), 1},
        {CoordBBox(Coord(3, 3, 0), Coord(5, 5, 2)), 1}});
    for (Conn conn : kAllConn) {
        expectTopo("solid", *solid, conn, 1, 0, 1);
        expectTopo("hollow", *hollow, conn, 1, 1, 2);
        expectTopo("torus", *torus, conn, 1, 0, 0);
    }
    // corner contact: joined only under 26
    expectTopo("corner", *corner, Conn::FG26_BG6, 1, 0, 1);
    expectTopo("corner", *corner, Conn::FG18_BG6, 2, 0, 2);
    expectTopo("corner", *corner, Conn::FG6_BG26, 2, 0, 2);
    expectTopo("corner", *corner, Conn::FG6_BG18, 2, 0, 2);
    // edge contact: joined under 26 and 18, separated under 6
    expectTopo("edge", *edge, Conn::FG26_BG6, 1, 0, 1);
    expectTopo("edge", *edge, Conn::FG18_BG6, 1, 0, 1);
    expectTopo("edge", *edge, Conn::FG6_BG26, 2, 0, 2);
    expectTopo("edge", *edge, Conn::FG6_BG18, 2, 0, 2);
}

TEST_F(TestLevelSetMultiObject, ConnectivityRuleMatters)
{
    // Two blocks touching only at a corner: one object under FG26_BG6, two
    // under FG6_BG26. Each rule's numbers agree with that rule's mesh;
    // verifying under one rule against the other rule's numbers fails and
    // the message names the rule.
    lsmo_test::ScopedDebugDir debugDir; // keeps the failure dumps out of the default directory
    auto corner = MLS::fromLabelGrid(*boxLabels({{CoordBBox(Coord(0), Coord(2)), 1},
        {CoordBBox(Coord(3), Coord(5)), 1}}));
    const CoordBBox dom(Coord(-2), Coord(8));
    const lsmo_test::DigitalTopology joined{1, 0, 1}, split{2, 0, 2};
    const auto r26 = lsmo_test::verifyLabelTopology(*corner, 1, dom, Conn::FG26_BG6, "FG26_BG6",
        &joined);
    EXPECT_TOPOLOGY_CONSISTENT(r26);
    EXPECT_EQ(r26.mesh.components, 1);
    EXPECT_EQ(r26.mesh.euler, 2);
    const auto r6 = lsmo_test::verifyLabelTopology(*corner, 1, dom, Conn::FG6_BG26, "FG6_BG26",
        &split);
    EXPECT_TOPOLOGY_CONSISTENT(r6);
    EXPECT_EQ(r6.mesh.components, 2);
    EXPECT_EQ(r6.mesh.euler, 4);
    const auto bad6 = lsmo_test::verifyLabelTopology(*corner, 1, dom, Conn::FG6_BG26,
        "FG6_BG26-vs-joined", &joined);
    EXPECT_FALSE(bad6.ok);
    EXPECT_NE(bad6.message.find("expected"), std::string::npos) << bad6.message;
    EXPECT_NE(bad6.message.find("rule FG6_BG26"), std::string::npos) << bad6.message;
    const auto bad26 = lsmo_test::verifyLabelTopology(*corner, 1, dom, Conn::FG26_BG6,
        "FG26_BG6-vs-split", &split);
    EXPECT_FALSE(bad26.ok);
    EXPECT_NE(bad26.message.find("rule FG26_BG6"), std::string::npos) << bad26.message;
    // the evolver's rule is what connectivityOf reports
    Forces forces;
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*corner, forces);
    for (Conn conn : kAllConn) {
        lsmo.setConnectivity(conn);
        EXPECT_EQ(lsmo_test::connectivityOf(lsmo), conn);
    }
}

TEST_F(TestLevelSetMultiObject, SimplePointPreservesTopology)
{
    // Independent check of isSimplePoint against digitalTopology: toggling a
    // simple voxel of a random blob never changes components, cavities or
    // the Euler characteristic.
    using tools::lsmo_internal::isSimplePoint;
    using lsmo_test::digitalTopology;
    const CoordBBox dom(Coord(-1), Coord(9));
    std::mt19937 gen(42u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> pos(1, 7);
    Index64 toggles = 0, broken = 0, nonSimpleChanged = 0, nonSimple = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const Conn conn = kAllConn[trial % 4];
        Int32Grid::Ptr g = Int32Grid::create(0);
        auto acc = g->getAccessor();
        const double density = 0.3 + 0.4 * uni(gen);
        for (int k = 0; k <= 8; ++k)
            for (int j = 0; j <= 8; ++j)
                for (int i = 0; i <= 8; ++i) if (uni(gen) < density) acc.setValue(Coord(i, j, k), 1);
        auto topo = digitalTopology(*g, 1, dom, conn);
        for (int n = 0; n < 60; ++n) {
            const Coord p(pos(gen), pos(gen), pos(gen));
            uint32_t mask = 0;
            for (int c = -1; c <= 1; ++c)
                for (int b = -1; b <= 1; ++b)
                    for (int a = -1; a <= 1; ++a)
                        if (acc.getValue(p.offsetBy(a, b, c)) == 1) mask |= nbBit(a, b, c);
            const bool simple = isSimplePoint(mask, conn);
            acc.setValue(p, acc.getValue(p) == 1 ? 0 : 1);
            const auto after = digitalTopology(*g, 1, dom, conn);
            if (simple) {
                ++toggles;
                if (!(after == topo)) ++broken;
            } else {
                ++nonSimple;
                if (!(after == topo)) ++nonSimpleChanged;
            }
            topo = after;
        }
    }
    std::cout << "SimplePointPreservesTopology: " << toggles << " simple toggles, " << broken
              << " changed topology; " << nonSimple << " non-simple toggles, "
              << nonSimpleChanged << " changed topology" << std::endl;
    EXPECT_GT(toggles, Index64(200));
    EXPECT_EQ(broken, Index64(0));
    EXPECT_GT(nonSimpleChanged, Index64(0));
}

namespace {

// The three spheres of the cube test: radius 4, centers at x = 48.
const Coord kOrbitSpheres[3] = {Coord(48, 10, 12), Coord(48, 20, 28), Coord(48, 29, 15)};
// Angular speed about x (rad per unit time): one revolution over the run.
const double kOrbitOmega = 2.0 * math::pi<double>() / 84.0;

// Common initial centroid of the spheres: the point the orbit axis
// (parallel to x) passes through.
Vec3d
orbitCenter()
{
    Vec3d c(0.0);
    for (const Coord& p : kOrbitSpheres) c += p.asVec3d();
    return c / 3.0;
}

// @a p rotated about the orbit axis by kOrbitOmega * @a t.
Vec3d
orbitPosition(const Vec3d& p, double t)
{
    const Vec3d c = orbitCenter();
    const double a = kOrbitOmega * t, dy = p[1] - c[1], dz = p[2] - c[2];
    return Vec3d(p[0], c[1] + dy * std::cos(a) - dz * std::sin(a),
        c[2] + dy * std::sin(a) + dz * std::cos(a));
}

// The same rigid rotation as a matrix in the row-vector convention of
// Mat4d::transform: x' = (x - c) R + c.
math::Mat4d
orbitMotion(double t)
{
    const Vec3d c = orbitCenter();
    const double a = kOrbitOmega * t, ca = std::cos(a), sa = std::sin(a);
    math::Mat4d m = math::Mat4d::identity();
    m(1, 1) = ca; m(1, 2) = sa;
    m(2, 1) = -sa; m(2, 2) = ca;
    for (int j = 0; j < 3; ++j) {
        double v = c[j];
        for (int i = 0; i < 3; ++i) v -= c[i] * m(i, j);
        m(3, j) = v;
    }
    return m;
}

// Volume centroid of a signed grid with the sub-voxel inside fraction
// H(psi) = clamp(0.5 - psi/dx, 0, 1), over @a box (index space).
Vec3d
insideCentroid(const FloatGrid& sdf, const CoordBBox& box)
{
    const double dx = sdf.voxelSize()[0];
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    Vec3d sum(0.0);
    double w = 0.0;
    for (auto p = box.begin(); p; ++p) {
        const double h = std::min(1.0, std::max(0.0, 0.5 - double(acc.getValue(*p)) / dx));
        sum += sdf.indexToWorld(*p) * h;
        w += h;
    }
    return w > 0.0 ? sum / w : sum;
}

constexpr double kCubeDx = 1.0; // voxel size of the cube scenario

struct CubeRunResult
{
    std::vector<lsmo_test::DigitalTopology> topo; // cube, per chunk
    double shift = 0.0;      // centroid x shift of label 1 (diagnostic)
    double frontShift = 0.0; // leading-face shift along the centerline column (sub-voxel)
    double maxXShift = 0.0;  // change of max x over all label-1 voxels (diagnostic)
    bool changed = false;    // cube topology differed from (1, 0, 1) at some chunk
    double firstChange = -1.0;                // end time of the first changed chunk
    lsmo_test::DigitalTopology firstChangeTopo; // cube topology at that chunk
    double maxCentroidError[3] = {0, 0, 0}; // sphere centroid vs rotated center, max over chunks (dx)
    double minVolumeFraction[3] = {1e9, 1e9, 1e9}; // sphere volume / (4/3) pi 4^3, min over chunks
    int maxComponents[3] = {0, 0, 0}; // sphere components, max over chunks (diagnostic)
    Index64 belowFloor = 0; // active voxels with phi < minimumPhi*dx, summed over chunks
    int maxCavities[3] = {0, 0, 0};   // sphere cavities, max over chunks (diagnostic)
    MLS::Ptr state;          // final state
};

// Cube (label 1, side 40 voxels, [0,39]^3) advected along +x by
// 1.5 x (side + sphere diameter + 8) = 84 voxels through three balls of
// radius 4 (labels 2-4, kinematic) centered at x = 48 at distinct (y, z),
// which orbit together rigidly about the axis parallel to x through their
// common centroid at kOrbitOmega (one revolution over the run), in the
// plane orthogonal to the cube's motion. The evolver protects topology
// under @a rule; object topology is verified with that same rule
// (connectivityOf(evolver)). With @a protect, label 1 is
// topology-preserving and the per-chunk gates are asserted (cube (1,0,1),
// consistent meshes, validate, partition, sphere centroids within 1 dx of
// the rotated centers; the obstacles' claims on cube voxels pass the
// simple-point test and trapped cube voxels are wrapped outward); the worst
// centroid error, smallest volume and largest component and cavity counts
// per sphere are returned (the last three as diagnostics: a sphere can be
// dented where it exits the protected cube). Without
// @a protect (the control) the run stops at the first chunk where the
// cube's topology changed. @a tEnd < 0 runs the whole distance.
CubeRunResult
runCubeThroughSpheres(bool protect, Conn rule, int grainSize = 1, double tEnd = -1.0,
    double motion = kDefaultMotion)
{
    const int side = 40;
    const bool fullRun = !(tEnd > 0.0);
    const double distance = fullRun ? 1.5 * (side + 8.0 + 8.0) : tEnd;
    const double fullDistance = 1.5 * (side + 8.0 + 8.0);
    const CoordBBox domain(Coord(-8, -8, -8),
        Coord(side - 1 + int(fullDistance) + 8, side + 7, side + 7));
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(side - 1)).begin(); p; ++p) acc.setValue(*p, 1);
        for (int s = 0; s < 3; ++s) {
            const Coord& c = kOrbitSpheres[s];
            for (auto p = CoordBBox(c - Coord(4), c + Coord(4)).begin(); p; ++p) {
                if ((*p - c).asVec3i().lengthSqr() <= 16) acc.setValue(*p, 2 + s);
            }
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4}));
    const double dx = m->voxelSize();
    auto labelCentroidX = [&]() {
        double sum = 0.0, n = 0.0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            if (acc.getValue(*p) == 1) { sum += (*p)[0]; n += 1.0; }
        }
        return n > 0.0 ? sum / n : 0.0;
    };
    const double x0 = labelCentroidX();
    auto labelMaxX = [&]() {
        int mx = std::numeric_limits<int>::min();
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            if (acc.getValue(*p) == 1) mx = std::max(mx, (*p)[0]);
        }
        return mx;
    };
    const int front0 = labelMaxX();
    // Leading face along the cube's centerline column (y, z) = (20, 20),
    // next to its center (19.5, 19.5): the last sub-voxel zero crossing
    // (- to +) of the extracted psi_1 along x. The orbits stay clear of it:
    // the sphere centers circle the orbit axis through (19.67, 18.33) at
    // radii 9.7-11.6, so a sphere of radius 4 comes no closer than 5.7
    // voxels to the axis, and this column is 1.7 voxels from it.
    const Coord column(0, 20, 20);
    auto centerFront = [&]() {
        FloatGrid::Ptr sdf = m->extractSignedDistance(1);
        tree::ValueAccessor<const FloatTree> acc(sdf->tree());
        double front = std::numeric_limits<double>::quiet_NaN();
        for (int x = domain.min()[0]; x < domain.max()[0]; ++x) {
            const float a = acc.getValue(Coord(x, column[1], column[2]));
            const float b = acc.getValue(Coord(x + 1, column[1], column[2]));
            if (a < 0.0f && !(b < 0.0f)) front = (x + double(a) / double(a - b)) * dx;
        }
        return front;
    };
    {
        const Vec3d c = orbitCenter();
        for (const Coord& sc : kOrbitSpheres) {
            const double r = std::hypot(sc[1] - c[1], sc[2] - c[2]);
            const double dc = std::hypot(column[1] - c[1], column[2] - c[2]);
            EXPECT_GT(r - 4.0 - 1.0, dc) << "an orbit crosses the centerline column";
        }
    }
    const double centerFront0 = centerFront();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(0.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    lsmo.setGrainSize(grainSize);
    applyMotion(lsmo, motion);
    for (int s = 0; s < 3; ++s) {
        lsmo.setKinematicLabel(2 + s, tools::createLevelSetSphere<FloatGrid>(4.0f,
            Vec3f(kOrbitSpheres[s].asVec3d()), float(dx)), orbitMotion);
    }
    lsmo.setConnectivity(rule);
    if (protect) lsmo.setTopologyPreservingLabels({1});
    // the matrix motion is the analytic rotation
    for (double tt : {0.0, 10.0, 31.0}) {
        const Vec3d p = kOrbitSpheres[0].asVec3d();
        EXPECT_LT((orbitMotion(tt).transform(p) - orbitPosition(p, tt)).length(), 1e-9);
    }
    const Conn conn = lsmo_test::connectivityOf(lsmo);
    const std::string ruleName = lsmo_test::connectivityName(conn);

    CubeRunResult res;
    const lsmo_test::DigitalTopology ball{1, 0, 1};
    const std::string tag = std::string(protect ? "protected" : "control") + "/" + ruleName;
    const std::string log = "TopologyCube(" + tag + ")";
    auto chunkTag = [&](double t, const std::string& what) {
        std::ostringstream os;
        os << tag << "/t" << std::setw(3) << std::setfill('0') << int(std::lround(t)) << "/" << what;
        return os.str();
    };
    EXPECT_TOPOLOGY_CONSISTENT(lsmo_test::verifyLabelTopology(*m, 1, domain, conn,
        chunkTag(0.0, "cube"), &ball, lsmo_test::phiFloorOf(lsmo)));
    auto volume = [](const FloatGrid& g) {
        return g.tree().activeVoxelCount() > 0 ? tools::levelSetVolume(g) : 0.0;
    };
    const double exactVolume = 4.0 / 3.0 * math::pi<double>() * 64.0 * dx * dx * dx;
    const double chunk = 2.0;
    double t = 0.0;
    size_t steps = 0;
    const auto start = std::chrono::steady_clock::now();
    while (t < distance - 1e-9) {
        const double t1 = std::min(distance, t + chunk);
        steps += lsmo.advect(t, t1);
        t = t1;
        const double floor = lsmo_test::phiFloorOf(lsmo);
        const auto cube = lsmo_test::verifyLabelTopology(*m, 1, domain, conn, chunkTag(t, "cube"),
            protect ? &ball : nullptr, floor);
        EXPECT_TOPOLOGY_CONSISTENT(cube);
        // the phi floor holds on protected interfaces (protected runs); without
        // protection there is no floor and smaller phi values occur
        EXPECT_TRUE(lsmo.validateTopologyFloor()) << "t " << t;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            if (*it < float(lsmo.minimumPhi() * dx)) ++res.belowFloor;
        }
        res.topo.push_back(cube.region);
        std::ostringstream row;
        row << log << ": t " << t << " steps " << steps << " cube " << cube.region;
        if (cube.region != ball && !res.changed) {
            res.changed = true;
            res.firstChange = t;
            res.firstChangeTopo = cube.region;
        }
        if (!protect) {
            std::cout << row.str() << std::endl;
            if (res.changed) {
                std::cout << log << ": first topology change at t " << t << ": " << cube.message
                          << std::endl;
                break;
            }
            continue;
        }
        bool chunkOk = cube.ok;
        for (int s = 0; s < 3; ++s) {
            const int32_t l = 2 + s;
            // Spheres: consistent meshes and centroids are gated; their
            // components and volume are diagnostics only, since cube voxels
            // the protected cube cannot give up can dent a sphere where it
            // exits the cube (see setKinematicLabel).
            const std::string stag = chunkTag(t, "sphere" + std::to_string(l));
            const auto sphere = lsmo_test::verifyLabelTopology(*m, l, domain, conn, stag, nullptr,
                floor);
            EXPECT_TOPOLOGY_CONSISTENT(sphere);
            chunkOk = chunkOk && sphere.ok;
            res.maxComponents[s] = std::max(res.maxComponents[s], sphere.region.components);
            res.maxCavities[s] = std::max(res.maxCavities[s], sphere.region.cavities);
            FloatGrid::Ptr sdf = m->extractSignedDistance(l);
            const Vec3d expected = orbitPosition(kOrbitSpheres[s].asVec3d(), t);
            const CoordBBox box(Coord::round(expected) - Coord(12), Coord::round(expected) + Coord(12));
            const double err = (insideCentroid(*sdf, box) - expected).length() / dx;
            const double frac = volume(*sdf) / exactVolume;
            res.maxCentroidError[s] = std::max(res.maxCentroidError[s], err);
            res.minVolumeFraction[s] = std::min(res.minVolumeFraction[s], frac);
            EXPECT_LE(err, 1.0) << stag;
            chunkOk = chunkOk && err <= 1.0;
            row << " L" << l << " " << sphere.region << " err " << err << " vol " << frac;
        }
        std::cout << row.str() << std::endl;
        std::string msg;
        const bool valid = m->validate(&msg);
        EXPECT_TRUE(valid) << "t " << t << ": " << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*m, overlaps, badCover);
        EXPECT_EQ(overlaps, Index64(0)) << "t " << t;
        EXPECT_EQ(badCover, Index64(0)) << "t " << t;
        // stop at this run's first failed chunk (not at failures of earlier runs)
        if (!(chunkOk && valid && overlaps == 0 && badCover == 0)) break;
    }
    res.shift = labelCentroidX() - x0;
    res.frontShift = centerFront() - centerFront0;
    res.maxXShift = double(labelMaxX() - front0);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << log << ": motion " << lsmo.maxInterfaceMotion() << ", " << steps << " steps, "
              << secs << " s (" << 1e3 * secs / double(std::max(steps, size_t(1)))
              << " ms/step), t " << t
              << ", active phi below minimumPhi*dx (summed over chunks) " << res.belowFloor
              << ", leading-face shift (centerline) " << res.frontShift
              << ", max-x shift (diagnostic) " << res.maxXShift << ", centroid shift " << res.shift
              << std::endl;
    if (protect) {
        for (int s = 0; s < 3; ++s) {
            std::cout << log << ": sphere " << 2 + s << " max centroid error "
                      << res.maxCentroidError[s] << " dx, min volume "
                      << 100.0 * res.minVolumeFraction[s] << "% of (4/3) pi 4^3, max components "
                      << res.maxComponents[s] << ", max cavities " << res.maxCavities[s]
                      << " (diagnostics)" << std::endl;
        }
    }
    res.state = m;
    return res;
}

} // anonymous namespace

namespace {

// The TopologyCubeThroughSpheres gates under both rules; @a motion as for
// runPressureGrowth.
void
runTopologyCubeGates(double motion)
{
    const double distance = 84.0;
    for (Conn rule : {Conn::FG26_BG6, Conn::FG6_BG26}) {
        SCOPED_TRACE(lsmo_test::connectivityName(rule));
        const CubeRunResult prot = runCubeThroughSpheres(true, rule, 1, -1.0, motion);
        EXPECT_FALSE(prot.changed);
        // Leading face along the centerline column, not the centroid:
        // topology protection leaves label-1 material tethered around the
        // spheres the cube has passed, so the centroid lags (printed only);
        // the max x over all label-1 voxels also sees material the wrap
        // left ahead of the face in a sphere's column (printed only).
        EXPECT_NEAR(prot.frontShift, distance, 2.0 * kCubeDx);
        for (int s = 0; s < 3; ++s) {
            // also asserted at every chunk; volume and components are printed only
            EXPECT_LE(prot.maxCentroidError[s], 1.0) << "sphere " << 2 + s;
        }
        // control: without protection the spheres change the cube's topology
        const CubeRunResult ctrl = runCubeThroughSpheres(false, rule, 1, -1.0, motion);
        EXPECT_TRUE(ctrl.changed);
        // no protected label: no floor, and phi values below it occur
        EXPECT_GT(ctrl.belowFloor, Index64(0));
        std::cout << "TopologyCube(control/" << lsmo_test::connectivityName(rule)
                  << "): first change at t " << ctrl.firstChange << " " << ctrl.firstChangeTopo
                  << std::endl;
    }
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, TopologyCubeThroughSpheres)
{
    runTopologyCubeGates(kDefaultMotion);
}

TEST_F(TestLevelSetMultiObject, Rk2StagesRespectProtection)
{
    // Every TVD-RK2 stage is a full update (projection with passive label 0
    // and fixed walls, topology protection, phi floor): with the default
    // schemes a protected cube keeps its topology through the kinematic
    // spheres (TopologySerialMatchesParallel checks the first kSerialCheckEnd
    // of the path under both rules, TopologyCubeThroughSpheres the whole
    // path), and a fixed label never changes (FixedLabels).
    {
        auto m = makeSphere(8.0);
        Forces forces;
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        ASSERT_EQ(lsmo.spatialScheme(), math::HJWENO5_BIAS);
        ASSERT_EQ(lsmo.temporalScheme(), math::TVD_RK2);
    }
    // fixed label 2 next to growing label 1
    auto m = makeTwoSpheres(1.0);
    std::map<Coord, float> before;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it == 2) before[it.getCoord()] = m->phi()->tree().getValue(it.getCoord());
    }
    ConstantSpeed speed;
    speed.s = -1.0f;
    speed.fixed = {2};
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
    for (int n = 0; n < 30; ++n) {
        lsmo.step(0.75f);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
    }
    Index64 changed = 0;
    for (const auto& kv : before) {
        const float phi = m->phi()->tree().getValue(kv.first);
        if (m->labels()->tree().getValue(kv.first) != 2
            || std::memcmp(&phi, &kv.second, sizeof(float)) != 0) ++changed;
    }
    EXPECT_EQ(changed, Index64(0));
}


namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

template<typename T>
void
fnvMix(uint64_t& h, const T& v)
{
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, &v, sizeof(T));
    for (unsigned char b : bytes) { h ^= uint64_t(b); h *= kFnvPrime; }
}

// HashValueT is the type each value is hashed as (the tree's own by default).
template<typename TreeT, typename HashValueT = typename TreeT::ValueType>
void
fnvMixTree(uint64_t& h, const TreeT& tree)
{
    for (auto it = tree.cbeginValueAll(); it; ++it) {
        const Coord p = it.getCoord();
        fnvMix(h, p.x());
        fnvMix(h, p.y());
        fnvMix(h, p.z());
        fnvMix(h, HashValueT(*it));
        fnvMix(h, uint8_t(it.isValueOn()));
        fnvMix(h, it.getVoxelCount());
    }
}

// FNV-1a hash of every value (coordinate, bits, activity, voxel count) of
// the phi and label trees of @a m. Labels are hashed as int32_t, so the hash
// does not depend on the label tree's value type (the recorded hashes date
// from 32-bit label trees).
uint64_t
stateHash(const MLS& m)
{
    uint64_t h = kFnvOffsetBasis;
    fnvMixTree(h, m.phi()->tree());
    fnvMixTree<LabelTree, int32_t>(h, m.labels()->tree());
    return h;
}

// Steps of the recorded partition hash, and of the grain-size check run
// when the hash is not compared (the serial run is the costly one).
constexpr int kPartitionSteps = 75, kPartitionGrainSteps = 25;

// The PartitionInvariant scenario (random constant speeds on a 4-label
// checker, @a steps steps) at grain size @a grainSize.
MLS::Ptr
runPartitionScenario(int grainSize, int steps)
{
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(31)).begin(); p; ++p) {
            acc.setValue(*p, 1 + (((*p)[0] / 8 + (*p)[1] / 8 + (*p)[2] / 8) % 4));
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    RandomSpeed speed({1, 2, 3, 4}, 20261108u);
    lsmo_frames::FrameEvolver<MLS, RandomSpeed> lsmo(*m, speed);
    lsmo.setGrainSize(grainSize);
    for (int n = 0; n < steps; ++n) lsmo.step(1.0f);
    return m;
}

// End time of the serial vs parallel cube checks and of the recorded cube
// hashes: the control cube (unprotected) first changes topology at t = 32
// for FG26_BG6 and t = 34 for FG6_BG26 in TopologyCubeThroughSpheres, and
// the spheres enter the cube from about t = 5, so 40 is well past contact,
// with the protected claim and wrap passes active.
constexpr double kSerialCheckEnd = 40.0;

// The protected runCubeThroughSpheres scenario under @a rule, evolution
// only (same setup and chunks of 2, no per-chunk checks), to @a tEnd.
MLS::Ptr
evolveCubeThroughSpheres(Conn rule, int grainSize, double tEnd)
{
    const int side = 40;
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(side - 1)).begin(); p; ++p) acc.setValue(*p, 1);
        for (int s = 0; s < 3; ++s) {
            const Coord& c = kOrbitSpheres[s];
            for (auto p = CoordBBox(c - Coord(4), c + Coord(4)).begin(); p; ++p) {
                if ((*p - c).asVec3i().lengthSqr() <= 16) acc.setValue(*p, 2 + s);
            }
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(0.0f);
    lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
    lsmo.setGrainSize(grainSize);
    for (int s = 0; s < 3; ++s) {
        lsmo.setKinematicLabel(2 + s, tools::createLevelSetSphere<FloatGrid>(4.0f,
            Vec3f(kOrbitSpheres[s].asVec3d()), float(m->voxelSize())), orbitMotion);
    }
    lsmo.setConnectivity(rule);
    lsmo.setTopologyPreservingLabels({1});
    for (double t = 0.0; t < tEnd - 1e-9;) {
        const double t1 = std::min(tEnd, t + 2.0);
        lsmo.advect(t, t1);
        t = t1;
    }
    return m;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, TopologySerialMatchesParallel)
{
    // The colored passes make the protected result independent of the
    // thread count: runs of the protected cube past its first contact with
    // the spheres are bit-identical serially and at grain sizes 1 and 64
    // under both rules (a hash of every value and activity bit, and a
    // voxel-by-voxel count). The grain-1 runs also assert the per-chunk gates of
    // runCubeThroughSpheres with the default TVD-RK2
    // (Rk2StagesRespectProtection).
    for (Conn rule : {Conn::FG26_BG6, Conn::FG6_BG26}) {
        SCOPED_TRACE(lsmo_test::connectivityName(rule));
        const CubeRunResult checked = runCubeThroughSpheres(true, rule, 1, kSerialCheckEnd);
        EXPECT_FALSE(checked.changed);
        EXPECT_NEAR(checked.frontShift, kSerialCheckEnd, 2.0 * kCubeDx);
        const MLS::Ptr a = checked.state;
        ASSERT_TRUE(a);
        std::vector<std::pair<int, MLS::Ptr>> others;
        others.emplace_back(0, evolveCubeThroughSpheres(rule, 0, kSerialCheckEnd));
        others.emplace_back(64, evolveCubeThroughSpheres(rule, 64, kSerialCheckEnd));
        for (const auto& [grain, b] : others) {
            SCOPED_TRACE("grain " + std::to_string(grain));
            ASSERT_TRUE(b);
            EXPECT_EQ(stateHash(*a), stateHash(*b));
            const LabelTree& la = a->labels()->tree();
            const LabelTree& lb = b->labels()->tree();
            const FloatTree& pa = a->phi()->tree();
            const FloatTree& pb = b->phi()->tree();
            ASSERT_TRUE(la.hasSameTopology(lb));
            Index64 diff = 0;
            for (auto it = pa.cbeginValueOn(); it; ++it) {
                const float v = pb.getValue(it.getCoord());
                if (std::memcmp(&v, &*it, sizeof(float)) != 0) ++diff;
            }
            for (auto it = la.cbeginValueAll(); it; ++it) {
                if (lb.getValue(it.getCoord()) != *it) ++diff;
            }
            EXPECT_EQ(diff, Index64(0));
        }
    }
}

TEST_F(TestLevelSetMultiObject, HashedContainersKeepResults)
{
    // Hashes of the phi and label trees recorded with the ordered containers
    // (std::map/std::set) the evolver used on its hot paths before they
    // became hashed ones: the switch must not change any result, for any
    // grain size. (A legitimate numerical change elsewhere re-records them.)
    // (re-recorded for the HJWENO5 / TVD-RK2 defaults, the vortex for the
    // speed extension and the inward-spiral scene, and the cube at
    // kSerialCheckEnd; the cube and
    // partition fields are uniform.) The hashes are bit-exact floating
    // point results of this platform's compiler and libm (FMA contraction
    // and math library differences change them), so they are compared only
    // with OPENVDB_MOGAC_CHECK_HASHES=1; LSMO_RECORD_HASHES prints them.
    // Grain independence is always asserted (here for the partition,
    // TopologySerialMatchesParallel for the cube).
    constexpr uint64_t kVortexHash = 0x8467808bb1d01dd1ull;
    constexpr uint64_t kCubeHash[2] = {0xda6e69f03bfe6e26ull, 0xa984ae7bcc6bda5eull}; // FG26, FG6
    constexpr uint64_t kPartitionHash = 0x1ce11d129535607eull;
    const bool record = std::getenv("LSMO_RECORD_HASHES") != nullptr;
    const char* checkEnv = std::getenv("OPENVDB_MOGAC_CHECK_HASHES");
    const bool committed = record || (checkEnv && std::string(checkEnv) == "1");
    auto check = [&](const char* what, uint64_t h, uint64_t expected) {
        if (record) {
            std::cout << "HashedContainers: " << what << " 0x" << std::hex << h << std::dec
                      << std::endl;
        } else {
            EXPECT_EQ(h, expected) << what;
        }
    };

    // the partition at every grain size (always asserted; over the first
    // kPartitionGrainSteps steps unless the hash is compared)
    uint64_t partition[3] = {0, 0, 0};
    const int grains[3] = {0, 1, 64};
    for (int g = 0; g < 3; ++g) {
        partition[g] = stateHash(*runPartitionScenario(grains[g],
            committed ? kPartitionSteps : kPartitionGrainSteps));
    }
    EXPECT_EQ(partition[0], partition[1]);
    EXPECT_EQ(partition[0], partition[2]);
    if (!committed) {
        std::cout << "HashedContainers: committed hashes not compared (set "
                     "OPENVDB_MOGAC_CHECK_HASHES=1)" << std::endl;
        return;
    }
    for (int g = 0; g < 3; ++g) {
        check(("partition grain " + std::to_string(grains[g])).c_str(), partition[g],
            kPartitionHash);
    }

    // label grid memory (informational): the 16-bit labels against the same
    // labels in a 32-bit grid
    auto printLabelMemory = [](const std::string& what, const MLS& m) {
        Int32Grid::Ptr wide = Int32Grid::create(
            Int32Tree::Ptr(new Int32Tree(m.labels()->tree())));
        wide->setTransform(m.labels()->transform().copy());
        for (auto meta = m.labels()->beginMeta(); meta != m.labels()->endMeta(); ++meta) {
            wide->insertMeta(meta->first, *meta->second);
        }
        std::cout << "LabelMemory: " << what << " labels()->memUsage() int16 "
                  << m.labels()->memUsage() << " B, as int32 " << wide->memUsage() << " B"
                  << std::endl;
    };

    const MLS::Ptr vortex = Vortex().endState();
    printLabelMemory("vortex", *vortex);
    check("vortex", stateHash(*vortex), kVortexHash);

    const Conn rules[2] = {Conn::FG26_BG6, Conn::FG6_BG26};
    for (int r = 0; r < 2; ++r) {
        // protected cube through the kinematic spheres, past first contact
        // (grain independent: TopologySerialMatchesParallel)
        const std::string name = "cube " + lsmo_test::connectivityName(rules[r]);
        const MLS::Ptr cube = evolveCubeThroughSpheres(rules[r], 64, kSerialCheckEnd);
        printLabelMemory(name, *cube);
        check(name.c_str(), stateHash(*cube), kCubeHash[r]);
    }
}

// ---------------------------------------------------------------------------
// Accuracy of extractSignedDistance after evolution, and world units
// ---------------------------------------------------------------------------

namespace {

// Self-consistency of extractSignedDistance(@a label): the extracted grid's
// zero isosurface is meshed (volumeToMesh) and converted back to an exact
// distance to that mesh (meshToLevelSet); @a ifaceMax is the max error over
// the voxels within one voxel of the surface, @a rms the RMS error over the
// extracted band, both in voxels. The metric's own floor (an analytic sphere
// against its marching-cubes mesh) is about 0.055*dx near the surface.
void
extractSelfConsistency(const MLS& m, int32_t label, double& ifaceMax, double& rms)
{
    FloatGrid::Ptr g = m.extractSignedDistance(label);
    std::vector<Vec3s> points;
    std::vector<Vec3I> tris;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(*g, points, tris, quads, 0.0);
    FloatGrid::Ptr ref = tools::meshToLevelSet<FloatGrid>(g->transform(), points, tris, quads,
        4.0f);
    const double dx = m.voxelSize();
    tree::ValueAccessor<const FloatTree> acc(ref->tree());
    ifaceMax = 0.0;
    double sum2 = 0.0;
    size_t n = 0;
    for (auto it = g->tree().cbeginValueOn(); it; ++it) {
        float r;
        if (!acc.probeValue(it.getCoord(), r)) continue;
        const double err = std::abs(double(*it) - double(r)) / dx;
        if (std::abs(r) < dx) ifaceMax = std::max(ifaceMax, err);
        sum2 += err * err;
        ++n;
    }
    rms = n ? std::sqrt(sum2 / double(n)) : 0.0;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, ExtractAccuracyAfterEvolution)
{
    // The signed distance stays accurate near the boundary after evolution:
    // max error < 0.1*dx within one voxel of the surface and RMS < 0.1*dx
    // over the band, against the distance to the extracted grid's own zero
    // isosurface, after pressure growth, uniform advection and curvature
    // flow of an off-grid sphere.
    struct Case { const char* name; double t1; std::function<void(Forces&)> setup; };
    const Case cases[] = {
        {"pressure growth", 6.0,
            [](Forces& f) { f.setPressure(FloatGrid::create(1.0f), 1.0f); }},
        {"uniform advection", 15.0,
            [](Forces& f) {
                f.setAdvection(Vec3SGrid::create(Vec3s(0.6f, 0.3f, 0.2f)), 1.0f);
            }},
        {"curvature flow", 7.5, [](Forces& f) { f.setCurvatureWeight(1.0f); }}};
    for (const Case& c : cases) {
        auto m = makeSphere(12.3, Vec3d(0.3, 0.1, 0.2));
        double iface0, rms0;
        extractSelfConsistency(*m, 1, iface0, rms0);
        Forces forces;
        c.setup(forces);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        const size_t steps = lsmo.advect(0.0, c.t1);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << c.name << ": " << msg;
        double iface, rms;
        extractSelfConsistency(*m, 1, iface, rms);
        std::cout << "ExtractAccuracyAfterEvolution " << c.name << " (t = " << c.t1 << ", "
                  << steps << " steps): interface max error " << iface << " dx, band RMS "
                  << rms << " dx (initial " << iface0 << " dx, " << rms0 << " dx)"
                  << std::endl;
        EXPECT_LE(iface, 0.1) << c.name;
        EXPECT_LE(rms, 0.1) << c.name;
    }
}

namespace {

// Shear rate of the velocity v = (kShearRate*z, 0, 0) of SpeedExtensionKeepsDistance.
constexpr float kShearRate = 0.05f;
// Gates of SpeedExtensionKeepsDistance: the extension-on interface max error
// is at most this fraction of the extension-off error, and at most this many
// voxels.
constexpr double kSpeedExtensionErrorRatio = 0.6;
constexpr double kSpeedExtensionMaxIfaceError = 0.15;

// The shear field on @a xf, dense over the region the band of the sheared
// sphere of SpeedExtensionKeepsDistance sweeps (zero outside).
Vec3SGrid::Ptr
makeShearField(const math::Transform& xf)
{
    Vec3SGrid::Ptr g = Vec3SGrid::create(Vec3f(0.0f));
    g->setTransform(xf.copy());
    auto acc = g->getAccessor();
    for (auto p = CoordBBox(Coord(-36, -20, -20), Coord(36, 20, 20)).begin(); p; ++p) {
        const Vec3d w = xf.indexToWorld(*p);
        acc.setValue(*p, Vec3f(kShearRate * float(w.z()), 0.0f, 0.0f));
    }
    return g;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, SpeedExtensionKeepsDistance)
{
    // ExtractAccuracyAfterEvolution's advection case in a shear field: the
    // closest-interface-point sampling of the velocity (the default) keeps
    // the distance property within kSpeedExtensionMaxIfaceError*dx at the
    // interface and clearly better than voxel-center sampling (at most
    // kSpeedExtensionErrorRatio of its error). The voxel-center error itself
    // is only reported (about 0.21*dx: the shear changes |grad psi| at the
    // interface voxels, which the redistancing keeps as seeds).
    {
        Forces f;
        EXPECT_TRUE(f.speedExtension()); // on by default
        f.setSpeedExtension(false);
        EXPECT_FALSE(f.speedExtension());
    }

    // The closest point of an interface voxel lies on the sphere, and is the
    // same for both labels of the interface (one sample serves both).
    {
        const double r = 12.3;
        const Vec3d c(0.3, 0.1, 0.2);
        auto m = makeSphere(r, c);
        Stencil s(*m);
        int n = 0;
        double worst = 0.0;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            if (std::abs(*it) > 1.0f) continue;
            const Coord p = it.getCoord();
            s.moveTo(p, 1);
            const Vec3d q1 = s.closestInterfacePoint();
            s.moveTo(p, 0);
            const Vec3d q0 = s.closestInterfacePoint();
            EXPECT_EQ(q0, q1) << p;
            worst = std::max(worst, std::abs((q1 - c).length() - r));
            ++n;
        }
        ASSERT_GT(n, 1000);
        std::cout << "SpeedExtensionKeepsDistance: closest points of " << n
                  << " interface voxels within " << worst << " dx of the sphere" << std::endl;
        EXPECT_LT(worst, 0.1);
    }

    constexpr double kT1 = 15.0;
    double iface[2], rms[2];
    for (int on = 0; on < 2; ++on) {
        auto m = makeSphere(12.3, Vec3d(0.3, 0.1, 0.2));
        Forces forces;
        forces.setSpeedExtension(on == 1);
        forces.setAdvection(makeShearField(m->phi()->transform()), 1.0f);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        const size_t steps = lsmo.advect(0.0, kT1);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        extractSelfConsistency(*m, 1, iface[on], rms[on]);
        std::cout << "SpeedExtensionKeepsDistance shear (t = " << kT1 << ", " << steps
                  << " steps, extension " << (on ? "on" : "off") << "): interface max error "
                  << iface[on] << " dx, band RMS " << rms[on] << " dx" << std::endl;
    }
    EXPECT_LE(iface[1], kSpeedExtensionErrorRatio * iface[0]);
    EXPECT_LE(iface[1], kSpeedExtensionMaxIfaceError);
}

TEST_F(TestLevelSetMultiObject, WorldUnitsHalfVoxel)
{
    // The PressureGrowth, UniformAdvection and CurvatureFlow scenarios with
    // the built-in forces at dx = 0.5: speeds, time steps and results are in
    // world units, so the same world-space gates as at dx = 1 hold.
    const double dx = 0.5;
    {
        auto m = makeSphere(8.0, Vec3d(0.0), dx);
        ASSERT_DOUBLE_EQ(m->voxelSize(), dx);
        Forces forces;
        forces.setPressure(FloatGrid::create(1.0f), 1.0f); // outward at unit speed
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        const size_t steps = lsmo.advect(0.0, 6.0);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
        std::cout << "WorldUnitsHalfVoxel pressure growth: steps " << steps
                  << " mean radius " << r << " (expected 14)" << std::endl;
        EXPECT_GE(steps, size_t(24)); // dt <= 0.5*cfl*dx/|F|
        EXPECT_NEAR(r, 8.0 + 6.0, 0.5);
    }
    {
        auto m = makeSphere(8.0, Vec3d(0.0), dx);
        Forces forces;
        forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        const Vec3d c0 = centroid(zeroCrossings(*m->extractSignedDistance(1)));
        const size_t steps = lsmo.advect(0.0, 7.5);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const auto pts = zeroCrossings(*m->extractSignedDistance(1));
        const Vec3d c1 = centroid(pts);
        std::cout << "WorldUnitsHalfVoxel uniform advection: steps " << steps << " shift "
                  << (c1 - c0) << " mean radius " << meanRadius(pts, c1) << std::endl;
        EXPECT_NEAR(c1[0] - c0[0], 7.5, 0.5);
        EXPECT_NEAR(c1[1] - c0[1], 0.0, 0.5);
        EXPECT_NEAR(c1[2] - c0[2], 0.0, 0.5);
        EXPECT_NEAR(meanRadius(pts, c1), 8.0, 0.5);
    }
    {
        auto m = makeSphere(10.0, Vec3d(0.0), dx);
        Forces forces;
        forces.setCurvatureWeight(1.0f);
        EXPECT_FLOAT_EQ(forces.stableDt(float(dx)), float(dx * dx / 6.0));
        lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
        const size_t steps = lsmo.advect(0.0, 30.0);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
        // r^2 = r0^2 - 2t = 100 - 60
        std::cout << "WorldUnitsHalfVoxel curvature flow: steps " << steps << " r^2 " << r * r
                  << " (expected 40)" << std::endl;
        EXPECT_NEAR(r * r, 40.0, 2.0 * std::sqrt(40.0) * 0.5);
    }
}


////////////////////////////////////////
// Incremental (dirty-region) redistancing

namespace {

// Advance @a e from @a t0 to @a t1 as advect() does, summing the redistance
// iterations of its steps into @a iters; returns the number of steps.
template<typename EvolverT>
size_t
advectCounting(EvolverT& e, double t0, double t1, size_t& iters)
{
    size_t n = 0;
    double t = t0;
    const double eps = 1e-6 * std::max(1.0, std::abs(t1));
    while (t1 - t > eps) {
        e.setTime(t);
        const float dt = e.step(float(t1 - t));
        if (!(dt > 0.0f)) break;
        iters += size_t(e.redistanceIterations());
        t += dt;
        ++n;
    }
    return n;
}

// Runs the same evolution twice from @a make(), with setFullRedistance(true)
// and (false), in chunks of @a chunk up to @a tEnd; after every chunk the
// label trees must be identical and max |phi_full - phi_incr| over the
// active voxels at most 1e-4*dx, and no redistance may stop at the
// iteration cap; with @a check false these are only printed. Prints the
// wall time and mean redistance iterations per step of both runs.
template<typename SpeedT, typename MakeT, typename ConfigT>
void
expectIncrementalMatchesFull(const std::string& name, const MakeT& make, const SpeedT& speed,
    const ConfigT& config, double tEnd, double chunk, bool check = true)
{
    using Evolver = lsmo_frames::FrameEvolver<MLS, SpeedT>;
    MLS::Ptr full = make(), incr = make();
    const double dx = full->voxelSize();
    // the default HJWENO5 needs halfWidth >= 3: narrower bands run FIRST_BIAS
    const math::BiasedGradientScheme spatial =
        full->halfWidth() < 3 ? math::FIRST_BIAS : math::HJWENO5_BIAS;
    Evolver ef(*full, speed, spatial), ei(*incr, speed, spatial);
    config(ef);
    config(ei);
    ef.setFullRedistance(true);
    ei.setFullRedistance(false);
    double secs[2] = {0.0, 0.0}, worst = 0.0;
    size_t steps = 0, iters[2] = {0, 0};
    bool labelsAgree = true;
    for (double t = 0.0; t < tEnd - 1e-9;) {
        const double t1 = std::min(tEnd, t + chunk);
        auto start = std::chrono::steady_clock::now();
        steps += advectCounting(ef, t, t1, iters[0]);
        secs[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        advectCounting(ei, t, t1, iters[1]);
        secs[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        t = t1;
        const Index64 labelDiff = treeDifferences(full->labels()->tree(), incr->labels()->tree());
        const bool sameTopology = full->phi()->tree().hasSameTopology(incr->phi()->tree());
        labelsAgree = labelsAgree && labelDiff == 0 && sameTopology;
        if (check) {
            EXPECT_EQ(labelDiff, Index64(0)) << name << " t " << t;
            EXPECT_TRUE(sameTopology) << name << " t " << t;
        }
        double maxDiff = 0.0;
        tree::ValueAccessor<const FloatTree> acc(incr->phi()->tree());
        for (auto it = full->phi()->tree().cbeginValueOn(); it; ++it) {
            float v;
            if (!acc.probeValue(it.getCoord(), v)) { maxDiff = 1e30; continue; }
            maxDiff = std::max(maxDiff, std::abs(double(*it) - double(v)));
        }
        worst = std::max(worst, maxDiff);
        if (check) {
            EXPECT_LE(maxDiff, 1e-4 * dx) << name << " t " << t;
            if (::testing::Test::HasFailure()) break;
        }
    }
    if (check) {
        EXPECT_EQ(ef.redistanceCapHits(), size_t(0)) << name;
        EXPECT_EQ(ei.redistanceCapHits(), size_t(0)) << name;
    }
    const double n = double(std::max<size_t>(steps, 1));
    std::cout << "IncrementalRedistance " << name << (check ? "" : " (not asserted)") << ": "
              << steps << " steps, max |phi_full - phi_incr| " << worst / dx << " dx, labels "
              << (labelsAgree ? "identical" : "DIFFER") << ", cap hits "
              << ef.redistanceCapHits() << "/" << ei.redistanceCapHits() << "; full "
              << 1e3 * secs[0] / n
              << " ms/step, " << double(iters[0]) / n << " iterations/step; incremental "
              << 1e3 * secs[1] / n << " ms/step, " << double(iters[1]) / n
              << " iterations/step" << std::endl;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, IncrementalRedistanceMatchesFull)
{
    auto none = [](auto&) {};
    for (int hw : {2, 3, 4, 5}) {
        const std::string h = "/hw" + std::to_string(hw);
        // agreement is asserted for halfWidth <= 3; wider bands are only
        // reported, over a shorter run ending at @a tShort (the converged
        // second-order Jacobi depends on the order in which voxels are
        // reached; see the class documentation)
        const bool check = hw <= 3;
        // (bands of halfWidth 2 run FIRST_BIAS, see expectIncrementalMatchesFull)
        auto compare = [&](const std::string& name, const auto& make, const auto& speed,
            const auto& config, double tEnd, double chunk, double tShort) {
            expectIncrementalMatchesFull(name + h, make, speed, config,
                check ? tEnd : std::min(tEnd, tShort), chunk, check);
        };
        auto sphere = [hw](float r) {
            std::map<int32_t, FloatGrid::ConstPtr> in;
            in[1] = tools::createLevelSetSphere<FloatGrid>(r, Vec3f(0.0f), 1.0f, float(hw));
            return MLS::fromSignedGrids(in, hw);
        };
        {
            // a growing sphere (label 1) next to a static one (label 2): most
            // of the band is clean, so kept values serve as Dirichlet data
            auto make = [hw] {
                std::map<int32_t, FloatGrid::ConstPtr> in;
                in[1] = tools::createLevelSetSphere<FloatGrid>(6.0f, Vec3f(0.0f), 1.0f,
                    float(hw));
                in[2] = tools::createLevelSetSphere<FloatGrid>(16.0f, Vec3f(30.0f, 0.0f, 0.0f),
                    1.0f, float(hw));
                return MLS::fromSignedGrids(in, hw);
            };
            LabelSpeed speed;
            speed.speeds = {{1, -1.0f}, {2, 0.0f}};
            compare("PartlyStatic", make, speed, none, 6.0, 1.0, 6.0);
        }
        {
            ConstantSpeed speed;
            speed.s = -1.0f;
            compare("PressureGrowth", [&] { return sphere(8.0f); },
                speed, none, 6.0, 1.0, 6.0);
        }
        {
            Forces forces;
            forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
            compare("UniformAdvection", [&] { return sphere(8.0f); },
                forces, none, 7.5, 1.5, 7.5);
        }
        {
            Forces forces;
            forces.setCurvatureWeight(1.0f);
            compare("CurvatureFlow", [&] { return sphere(10.0f); },
                forces, none, 30.0, 5.0, 10.0);
        }
        {
            const Vortex vortex;
            Forces forces;
            forces.setAdvection(vortex.velocity(), 1.0f);
            forces.setCurvatureWeight(float(Vortex::kCurvatureWeight));
            compare("Vortex",
                [&] { return MLS::fromSignedGrids(vortex.inputs(), hw); }, forces, none,
                Vortex::kEndTime, 2.0, 8.0);
        }
        {
            // TopologyCubeThroughSpheres: protected cube (label 1) advected
            // along +x through three orbiting kinematic spheres (labels 2-4)
            const int side = 40;
            auto make = [&] {
                Int32Grid::Ptr labels = Int32Grid::create(0);
                auto acc = labels->getAccessor();
                for (auto p = CoordBBox(Coord(0), Coord(side - 1)).begin(); p; ++p) {
                    acc.setValue(*p, 1);
                }
                for (int s = 0; s < 3; ++s) {
                    const Coord& c = kOrbitSpheres[s];
                    for (auto p = CoordBBox(c - Coord(4), c + Coord(4)).begin(); p; ++p) {
                        if ((*p - c).asVec3i().lengthSqr() <= 16) acc.setValue(*p, 2 + s);
                    }
                }
                return MLS::fromLabelGrid(*labels, hw);
            };
            Forces forces;
            forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
            forces.setCurvatureWeight(0.0f);
            auto config = [](lsmo_frames::FrameEvolver<MLS, Forces>& e) {
                for (int s = 0; s < 3; ++s) {
                    e.setKinematicLabel(2 + s, tools::createLevelSetSphere<FloatGrid>(4.0f,
                        Vec3f(kOrbitSpheres[s].asVec3d()), 1.0f), orbitMotion);
                }
                e.setConnectivity(Conn::FG26_BG6);
                e.setTopologyPreservingLabels({1});
            };
            compare("TopologyCubeThroughSpheres", make, forces, config,
                84.0, 2.0, 6.0);
        }
        {
            // Only kinematic writes change the pair: a static protected cube
            // (label 1, zero speeds) and a static ball (label 3) next to it; a
            // kinematic ball (label 2) moves along +y into the cube (claims,
            // rejected claims, wraps) and on past the static ball.
            auto make = [hw] {
                Int32Grid::Ptr labels = Int32Grid::create(0);
                auto acc = labels->getAccessor();
                for (auto p = CoordBBox(Coord(0), Coord(23)).begin(); p; ++p) acc.setValue(*p, 1);
                for (auto p = CoordBBox(Coord(-30, -20, 0), Coord(-10, 0, 20)).begin(); p; ++p) {
                    if ((*p - Coord(-20, -10, 10)).asVec3i().lengthSqr() <= 64) {
                        acc.setValue(*p, 3);
                    }
                }
                return MLS::fromLabelGrid(*labels, hw);
            };
            LabelSpeed speed; // all zero
            auto ball = [](double y0, double v) {
                return [y0, v](lsmo_frames::FrameEvolver<MLS, LabelSpeed>& e) {
                    e.setKinematicLabel(2, tools::createLevelSetSphere<FloatGrid>(4.0f,
                        Vec3f(12.0f, float(y0), 12.0f), 1.0f), [v](double t) {
                            math::Mat4d m = math::Mat4d::identity();
                            m.setTranslation(Vec3d(0.0, v * t, 0.0));
                            return m;
                        });
                    e.setTopologyPreservingLabels({1});
                };
            };
            compare("KinematicOnly", make, speed, ball(-8.0, 1.0),
                16.0, 2.0, 12.0);
            // the same ball moving slowly (one step per chunk, a hundredth of a
            // voxel per step), so that few voxels change their freeze state
            // and the stamped values carry the changes
            compare("KinematicSlow", make, speed, ball(-3.3, 0.04),
                6.0, 0.25, 3.0);
        }
    }
}


TEST_F(TestLevelSetMultiObject, MaxInterfaceMotionApi)
{
    auto m = makeSphere(8.0);
    ConstantSpeed speed;
    lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), tools::lsmo_internal::kDefaultMaxInterfaceMotionVoxels);
    lsmo.setMaxInterfaceMotion(tools::lsmo_internal::kFullVoxelMotion);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), 1.0);
    lsmo.setMaxInterfaceMotion(tools::lsmo_internal::kHalfVoxelMotion);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), 0.5);
    for (double bad : {0.0, -0.5, 0.75, 1.5, 2.0, std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(lsmo.setMaxInterfaceMotion(bad), ValueError) << bad;
        EXPECT_EQ(lsmo.maxInterfaceMotion(), 0.5) << bad;
    }
    // the cap of an unconstrained step: motion * cfl * dx / max|F|
    speed.s = -1.0f;
    EXPECT_FLOAT_EQ(lsmo.step(10.0f), 0.5f * lsmo.cfl());
    lsmo.setMaxInterfaceMotion(1.0);
    EXPECT_FLOAT_EQ(lsmo.step(10.0f), lsmo.cfl());
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, FullVoxelMotion)
{
    // Every gate of the half-voxel scenarios, unchanged, at full-voxel
    // motion (setMaxInterfaceMotion(1.0)): ContactNoCascade (label-2 extent
    // growth at most one voxel per step, no cascade), PressureGrowth,
    // UniformAdvection, CurvatureFlow, VortexEightObjects (t = 60),
    // TopologyCubeThroughSpheres (both rules, protected and control) and
    // FixedLabels; and PressureGrowth takes at most 0.6x the steps of the
    // half-voxel run.
    constexpr double kFull = tools::lsmo_internal::kFullVoxelMotion;
    constexpr double kHalf = tools::lsmo_internal::kHalfVoxelMotion;
    constexpr double kMaxStepRatio = 0.6;
    {
        SCOPED_TRACE("ContactNoCascade");
        runContactNoCascade(kFull);
    }
    {
        SCOPED_TRACE("PressureGrowth");
        const size_t half = runPressureGrowth(kHalf);
        const size_t full = runPressureGrowth(kFull);
        std::cout << "FullVoxelMotion: PressureGrowth steps " << full << " vs " << half
                  << " at half-voxel motion" << std::endl;
        EXPECT_LE(double(full), kMaxStepRatio * double(half));
    }
    {
        SCOPED_TRACE("UniformAdvection");
        runUniformAdvection(kFull);
    }
    {
        SCOPED_TRACE("CurvatureFlow");
        runCurvatureFlow(kFull);
    }
    {
        SCOPED_TRACE("FixedLabels");
        runFixedLabels(kFull);
    }
    {
        SCOPED_TRACE("VortexEightObjects");
        runVortexEightObjects(kFull);
    }
    {
        SCOPED_TRACE("TopologyCubeThroughSpheres");
        runTopologyCubeGates(kFull);
    }
}


namespace {

// Two spheres side by side (labels 1 and 2) and a small one (label 3)
// above them, half width 3.
MLS::Ptr
makeEditScene()
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(7.0f, Vec3f(-6.0f, 0.0f, 0.0f), 1.0f, 3.0f);
    in[2] = tools::createLevelSetSphere<FloatGrid>(7.0f, Vec3f(6.0f, 0.0f, 0.0f), 1.0f, 3.0f);
    in[3] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(0.0f, 10.0f, 0.0f), 1.0f, 3.0f);
    return MLS::fromSignedGrids(in, 3);
}

// External edits of @a m: dilate the band by one voxel and rename label
// @a from to @a to (every voxel and tile, active or not).
void
editExternally(MLS& m, int32_t from, int32_t to)
{
    m.dilate(1);
    using LabelT = LabelTree::ValueType;
    for (auto it = m.labels()->tree().beginValueAll(); it; ++it) {
        if (*it == LabelT(from)) it.setValue(LabelT(to));
    }
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, ExternalEditsMarked)
{
    // Edits of the pair between advect() calls (MultiLevelSet::dilate() and
    // a relabel) announced with markExternallyModified(): the next step
    // re-solves the whole band, so builds with OPENVDB_ENABLE_ASSERTS do not
    // abort in the dirty-region check, and the incremental run keeps
    // matching a full-redistance run (labels identical, phi within the
    // redistance tolerance, as in IncrementalRedistanceMatchesFull).
    using Evolver = lsmo_frames::FrameEvolver<MLS, Forces>;
    MLS::Ptr incr = makeEditScene(), full = makeEditScene();
    const double dx = incr->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(0.5f, 0.25f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(0.2f);
    Evolver ei(*incr, forces), ef(*full, forces);
    ef.setFullRedistance(true);
    constexpr double kChunk = 1.0;
    constexpr int kChunks = 6;
    for (int c = 0; c < kChunks; ++c) {
        SCOPED_TRACE("chunk " + std::to_string(c));
        if (c == 2 || c == 4) {
            // a relabel after the first edit, back again after the second
            const int32_t from = c == 2 ? 3 : 5, to = c == 2 ? 5 : 3;
            editExternally(*incr, from, to);
            editExternally(*full, from, to);
            ei.markExternallyModified();
            ef.markExternallyModified();
        }
        const double t0 = c * kChunk;
        EXPECT_GT(ei.advect(t0, t0 + kChunk), size_t(0));
        ef.advect(t0, t0 + kChunk);
        std::string msg;
        EXPECT_TRUE(incr->validate(&msg)) << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*incr, overlaps, badCover);
        EXPECT_EQ(overlaps, Index64(0));
        EXPECT_EQ(badCover, Index64(0));
        EXPECT_EQ(treeDifferences(full->labels()->tree(), incr->labels()->tree()), Index64(0));
        ASSERT_TRUE(full->phi()->tree().hasSameTopology(incr->phi()->tree()));
        double maxDiff = 0.0;
        tree::ValueAccessor<const FloatTree> acc(incr->phi()->tree());
        for (auto it = full->phi()->tree().cbeginValueOn(); it; ++it) {
            maxDiff = std::max(maxDiff, std::abs(double(*it) - double(acc.getValue(it.getCoord()))));
        }
        EXPECT_LE(maxDiff, 1e-4 * dx);
        const std::vector<int32_t> expected = (c >= 2 && c < 4)
            ? std::vector<int32_t>{1, 2, 5} : std::vector<int32_t>{1, 2, 3};
        EXPECT_EQ(incr->labelSet(), expected);
    }
}


namespace {

// Feature combination: a topology-protected hollow cube (label 1, one
// component, one cavity, holding label 4), a volume-constrained sphere
// (label 2) driven toward it by its own advection field, and a kinematic
// sphere (label 3) moving through the cube wall into the cavity, under a
// shared advection field and curvature.
MLS::Ptr
makeComboScene()
{
    constexpr int kCube = 18, kWall = 3;
    Int32Grid::Ptr labels = Int32Grid::create(0);
    auto acc = labels->getAccessor();
    for (auto p = CoordBBox(Coord(0), Coord(kCube - 1)).begin(); p; ++p) {
        const bool inner = CoordBBox(Coord(kWall), Coord(kCube - 1 - kWall)).isInside(*p);
        acc.setValue(*p, inner ? 4 : 1);
    }
    const Coord c2(29, 9, 9);
    for (auto p = CoordBBox(c2 - Coord(7), c2 + Coord(7)).begin(); p; ++p) {
        if ((*p - c2).asVec3d().length() < 6.0) acc.setValue(*p, 2);
    }
    return MLS::fromLabelGrid(*labels, 3);
}

// to t = 14: the kinematic sphere enters the cube wall at t = 6 and the
// cavity at about t = 9; the constrained sphere reaches the cube at about
// t = 8
constexpr double kComboEnd = 14.0, kComboChunk = 2.0;

// Configure @a e for the combination (curvature is in the forces).
template<typename EvolverT>
void
configureCombo(EvolverT& e, double motion)
{
    e.setMaxInterfaceMotion(motion);
    e.setTopologyPreservingLabels({1});
    e.preserveInitialVolume(2);
    e.setKinematicLabel(3, tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(0.0f), 1.0f, 3.0f),
        [](double t) {
            math::Mat4d m = math::Mat4d::identity();
            m.setTranslation(Vec3d(-10.0 + t, 9.0, 9.0));
            return m;
        });
}

Forces
comboForces(double curvature)
{
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(0.3f, 0.0f, 0.0f)), 1.0f);
    forces.setAdvection(2, Vec3SGrid::create(Vec3s(-0.8f, 0.0f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(float(curvature));
    return forces;
}

// Invariants of the combination after a chunk: validate, partition, phi
// floor, no active phi = 0, the protected cube keeps one component and one
// cavity.
template<typename EvolverT>
void
expectComboInvariants(const MLS& m, const EvolverT& e)
{
    // the cube (0..17, moving +x at 0.3) and the spheres near it
    const CoordBBox domain(Coord(-16), Coord(40));
    std::string msg;
    EXPECT_TRUE(m.validate(&msg)) << msg;
    Index64 overlaps = 0, badCover = 0;
    partitionErrors(m, overlaps, badCover);
    EXPECT_EQ(overlaps, Index64(0));
    EXPECT_EQ(badCover, Index64(0));
    EXPECT_TRUE(e.validateTopologyFloor());
    Index64 zeros = 0;
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) zeros += (*it == 0.0f);
    EXPECT_EQ(zeros, Index64(0));
    const auto cube = lsmo_test::digitalTopology(*m.labels(), 1, domain,
        tools::MultiObjectConnectivity::FG26_BG6);
    EXPECT_EQ(cube.components, 1);
    EXPECT_EQ(cube.cavities, 1);
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, FeatureCombination)
{
    // Topology preservation, a volume constraint, per-label advection, a
    // kinematic label and curvature together, with the default HJWENO5 /
    // TVD-RK2 schemes, at half- and full-voxel motion and at a weak (0.3)
    // and a strong (1.5, split steps) curvature. A serial run (incremental
    // redistance) is checked after every chunk (expectComboInvariants) and
    // against a full-redistance run (labels and topology identical, phi
    // within 1e-4*dx, halfWidth 3); a grain-64 run ends bit-identical to it.
    using Evolver = lsmo_frames::FrameEvolver<MLS, Forces>;
    for (double curvature : {0.3, 1.5}) {
        for (double motion : {0.5, 1.0}) {
            std::ostringstream os;
            os << "FeatureCombination/curvature " << curvature << "/motion " << motion;
            const std::string name = os.str();
            SCOPED_TRACE(name);
            const auto start = std::chrono::steady_clock::now();
            const Forces forces = comboForces(curvature);
            MLS::Ptr serial = makeComboScene(), parallel = makeComboScene(),
                full = makeComboScene();
            const double dx = serial->voxelSize();
            Evolver es(*serial, forces), ep(*parallel, forces), ef(*full, forces);
            es.setGrainSize(0);
            ep.setGrainSize(64);
            ef.setFullRedistance(true);
            for (Evolver* e : {&es, &ep, &ef}) configureCombo(*e, motion);
            size_t steps = 0;
            double worst = 0.0;
            for (double t = 0.0; t < kComboEnd - 1e-9; t += kComboChunk) {
                SCOPED_TRACE("t " + std::to_string(t + kComboChunk));
                const size_t n = es.advect(t, t + kComboChunk);
                EXPECT_GT(n, size_t(0));
                steps += n;
                ep.advect(t, t + kComboChunk);
                ef.advect(t, t + kComboChunk);
                expectComboInvariants(*serial, es);
                EXPECT_EQ(treeDifferences(full->labels()->tree(), serial->labels()->tree()),
                    Index64(0));
                ASSERT_TRUE(full->phi()->tree().hasSameTopology(serial->phi()->tree()));
                tree::ValueAccessor<const FloatTree> acc(serial->phi()->tree());
                for (auto it = full->phi()->tree().cbeginValueOn(); it; ++it) {
                    worst = std::max(worst,
                        std::abs(double(*it) - double(acc.getValue(it.getCoord()))));
                }
                EXPECT_LE(worst, 1e-4 * dx);
                if (::testing::Test::HasFailure()) break;
            }
            EXPECT_EQ(stateHash(*serial), stateHash(*parallel));
            EXPECT_EQ(es.redistanceCapHits(), size_t(0));
            EXPECT_EQ(ef.redistanceCapHits(), size_t(0));
            std::cout << name << ": " << steps << " steps, max |phi_full - phi_incr| "
                      << worst / dx << " dx, euler fallbacks " << es.eulerFallbacks() << ", "
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count()
                      << " s" << std::endl;
        }
    }
}


////////////////////////////////////////
// Step benchmark (informational: prints only, asserts nothing about time).

namespace {

struct BenchConfig
{
    const char* name;
    bool legacy;     // the pre-optimization settings
    double motion;   // 0: keep the default
};

struct BenchResult { size_t steps = 0; double secs = 0.0; };

template<typename EvolverT>
void
configureBench(EvolverT& e, const BenchConfig& c)
{
    if (c.legacy) {
        e.setSpatialScheme(math::FIRST_BIAS);
        e.setTemporalScheme(math::TVD_RK1);
        e.setSplitting(false);
        e.setFullRedistance(true);
        e.setMaxInterfaceMotion(0.5);
    }
    applyMotion(e, c.motion);
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, StepBenchmark)
{
    if (!std::getenv("OPENVDB_MOGAC_BENCHMARK")) {
        GTEST_SKIP() << "informational; set OPENVDB_MOGAC_BENCHMARK=1 to run";
    }
    // Advances each scene (Vortex to t = 20, before first contact; the sphere
    // growth to t = 10; shorter than the correctness runs) under three
    // configurations:
    // (a) the pre-optimization settings (FIRST_BIAS, TVD_RK1, no splitting,
    // full redistance, 0.5 motion, speed extension off), (b) the defaults,
    // (c) the defaults with setMaxInterfaceMotion(1.0). Prints ms/step,
    // steps, total time and speedups relative to (a).
    const BenchConfig configs[3] = {
        {"(a) pre-plan", true, 0.0}, {"(b) defaults", false, 0.0},
        {"(c) defaults + motion 1.0", false, 1.0}};
    for (int scene = 0; scene < 2; ++scene) {
        const char* sceneName = scene == 0 ? "Vortex" : "SphereGrowth";
        const double kEnd = scene == 0 ? 20.0 : 10.0;
        BenchResult res[3];
        Index64 active = 0;
        for (int k = 0; k < 3; ++k) {
            if (scene == 0) {
                const Vortex vortex;
                auto m = MLS::fromSignedGrids(vortex.inputs(), 3);
                active = m->phi()->activeVoxelCount();
                Forces forces;
                forces.setAdvection(vortex.velocity(), 1.0f);
                forces.setCurvatureWeight(float(vortex.curvature));
                if (configs[k].legacy) forces.setSpeedExtension(false);
                lsmo_frames::FrameEvolver<MLS, Forces> lsmo(*m, forces);
                configureBench(lsmo, configs[k]);
                const auto start = std::chrono::steady_clock::now();
                res[k].steps = lsmo.advect(0.0, kEnd);
                res[k].secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                std::string msg;
                EXPECT_TRUE(m->validate(&msg)) << msg;
            } else {
                auto m = makeSphere(100.0);
                active = m->phi()->activeVoxelCount();
                ConstantSpeed speed;
                speed.s = -1.0f;
                lsmo_frames::FrameEvolver<MLS, ConstantSpeed> lsmo(*m, speed);
                configureBench(lsmo, configs[k]);
                const auto start = std::chrono::steady_clock::now();
                res[k].steps = lsmo.advect(0.0, kEnd);
                res[k].secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                std::string msg;
                EXPECT_TRUE(m->validate(&msg)) << msg;
            }
            EXPECT_GT(res[k].steps, size_t(0));
        }
        std::cout << "StepBenchmark: " << sceneName << ", t = " << kEnd << ", " << active
                  << " active voxels at start" << std::endl;
        for (int k = 0; k < 3; ++k) {
            const double ms = 1e3 * res[k].secs / double(std::max(res[k].steps, size_t(1)));
            const double ms0 = 1e3 * res[0].secs / double(std::max(res[0].steps, size_t(1)));
            std::cout << "StepBenchmark:   " << configs[k].name << ": " << res[k].steps
                      << " steps, " << ms << " ms/step, " << res[k].secs << " s total, "
                      << "per-step speedup " << ms0 / ms << "x, total speedup "
                      << res[0].secs / res[k].secs << "x" << std::endl;
        }
    }
}
