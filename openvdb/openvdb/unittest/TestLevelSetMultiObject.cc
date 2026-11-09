// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
using MLS = tools::MultiLevelSet<>;

// 16^3 block: label 1 for x<8, 2 for x>=8, phi = |x - 7.5|*dx clamped to 3*dx.
MLS::Ptr
makePair(double dx = 1.0)
{
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3.0 * dx));
    Int32Grid::Ptr lab = Int32Grid::create(0);
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
    return MLS::Ptr(new MLS(phi, lab, 3));
}

} // anonymous namespace

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
    EXPECT_FALSE(msg.empty());

    auto m2 = makePair();
    m2->labels()->tree().setValueOn(Coord(40, 0, 0), 1);
    msg.clear();
    EXPECT_FALSE(m2->validate(&msg));
    EXPECT_FALSE(msg.empty());
}

TEST_F(TestLevelSetMultiObject, HolderThrows)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    Int32Grid::Ptr lab = Int32Grid::create(0);
    phi->setTransform(math::Transform::createLinearTransform(1.0));
    lab->setTransform(math::Transform::createLinearTransform(0.5));
    EXPECT_THROW(MLS(phi, lab, 3), ValueError);

    lab->setTransform(math::Transform::createLinearTransform(1.0));
    EXPECT_THROW(MLS(phi, lab, 1), ValueError);
    EXPECT_NO_THROW(MLS(phi, lab, 2));

    auto nonUniform = math::Transform::Ptr(new math::Transform(math::MapBase::Ptr(new math::ScaleMap(Vec3d(1.0, 2.0, 1.0)))));
    FloatGrid::Ptr phi2 = FloatGrid::create(3.0f);
    Int32Grid::Ptr lab2 = Int32Grid::create(0);
    phi2->setTransform(nonUniform);
    lab2->setTransform(nonUniform->copy());
    EXPECT_THROW(MLS(phi2, lab2, 3), ValueError);
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
    auto m = makePair();
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
    // new voxels outside x<0 copy label 1 from their neighbours
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
        Int32Grid::Ptr lab = Int32Grid::create(0);
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
    Int32Grid::Ptr lab = Int32Grid::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    const CoordBBox box(Coord(3, 5, 2), Coord(13, 12, 10)); // not leaf aligned
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    auto labelAt = [](const Coord& c) { return c[0] < 8 ? 1 : 2; };
    for (auto it = box.begin(); it; ++it) {
        pa.setValue(*it, 0.25f);
        la.setValue(*it, labelAt(*it));
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
// The interface layers hold the exact distances; if @a exact is false the
// outer band voxels hold garbage (alternating 0.5*dx and halfWidth*dx).
MLS::Ptr
makePlanar(double dx, bool exact)
{
    const int halfWidth = 3;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
    Int32Grid::Ptr lab = Int32Grid::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = 4; i <= 11; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const Coord c(i, j, k);
                float d = float(std::abs(i - 7.3) * dx);
                if (!exact && i != 7 && i != 8) {
                    d = ((i + j + k) % 2) ? float(0.5 * dx) : float(halfWidth * dx);
                }
                pa.setValue(c, d);
                la.setValue(c, i <= 7 ? 1 : 2);
            }
    // One inactive shell around the band holding the deep-interior labels
    // (ruling R7b), with phi leaves kept in lockstep at the background.
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
    auto m = makePlanar(dx, /*exact=*/false);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const Index64 nActive = m->phi()->activeVoxelCount();
    const Index64 nLeaves = m->phi()->tree().leafCount();

    tools::MultiLevelSetRedistance<MLS> redist(*m);
    const int iterations = redist.redistance();
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

TEST_F(TestLevelSetMultiObject, RedistanceKeepsCrossing)
{
    const double dx = 0.5;
    auto m = makePlanar(dx, /*exact=*/true);
    tools::MultiLevelSetRedistance<MLS> redist(*m);
    redist.redistance();
    const auto& t = m->phi()->tree();
    for (int j = 0; j < 16; ++j)
        for (int k = 0; k < 16; ++k) {
            const double a = t.getValue(Coord(7, j, k));
            const double b = t.getValue(Coord(8, j, k));
            ASSERT_GT(a + b, 0.0);
            EXPECT_NEAR(7.0 + a / (a + b), 7.3, 1e-5);
        }
}

TEST_F(TestLevelSetMultiObject, RedistanceOrderIndependent)
{
    // Three labels over several leaves: a ball split into labels 1 | 2 by the
    // plane x = 11.5, label 3 outside; phi starts as a rough guess.
    auto build = []() {
        const double dx = 0.25;
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(float(3 * dx));
        Int32Grid::Ptr lab = Int32Grid::create(0);
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        const Vec3d ctr(11.5, 12.2, 11.7);
        const double R = 7.4;
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                for (int k = 0; k < 24; ++k) {
                    const double r = (Vec3d(i, j, k) - ctr).length();
                    if (std::abs(r - R) > 4.0) continue;
                    const Coord c(i, j, k);
                    pa.setValue(c, float(std::abs(r - R) * dx * (1.0 + 0.1 * ((i * 7 + j * 3 + k) % 5))));
                    la.setValue(c, r < R ? (i < 12 ? 1 : 2) : 3);
                }
        return MLS::Ptr(new MLS(phi, lab, 3));
    };
    auto serial = build();
    auto parallel = build();
    ASSERT_GT(serial->phi()->tree().leafCount(), Index32(8));

    tools::MultiLevelSetRedistance<MLS> r0(*serial);
    r0.setGrainSize(0);
    const int n0 = r0.redistance();
    tools::MultiLevelSetRedistance<MLS> r1(*parallel);
    r1.setGrainSize(1);
    const int n1 = r1.redistance();
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

TEST_F(TestLevelSetMultiObject, RedistanceSeedInterface)
{
    const double dx = 0.5;
    auto m = makePlanar(dx, /*exact=*/false);
    const FloatTree snapshot(m->phi()->tree());
    FloatTree out(m->phi()->tree());
    BoolTree seeds(false);
    tools::MultiLevelSetRedistance<MLS>::seedInterface(snapshot,
        [](const Coord& c) { return c[0] <= 7; }, out, seeds, dx);
    EXPECT_EQ(seeds.activeVoxelCount(), Index64(2 * 16 * 16));
    for (auto it = seeds.cbeginValueOn(); it; ++it) {
        const Coord c = it.getCoord();
        ASSERT_TRUE(c[0] == 7 || c[0] == 8) << c;
        EXPECT_NEAR(out.getValue(c), std::abs(c[0] - 7.3) * dx, 1e-6);
    }
    // non-seed voxels are left alone
    EXPECT_EQ(out.getValue(Coord(5, 3, 3)), snapshot.getValue(Coord(5, 3, 3)));
}
