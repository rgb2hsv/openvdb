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
#include <openvdb/tools/VelocityFields.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
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
// Layers x in [7 - halo, 8 + halo] hold the exact distances (halo 0: the two
// interface layers, enough for redistance(); halo 1: also the outer
// neighbours read by the spec 5.1 reseed); if @a exact is false the remaining
// band voxels hold garbage (alternating 0.5*dx and halfWidth*dx).
MLS::Ptr
makePlanar(double dx, bool exact, int halo = 1)
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
                if (!exact && (i < 7 - halo || i > 8 + halo)) {
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
// box [lo, hi]^3 inactive with its deep-interior label (ruling R7b).
MLS::Ptr
makeBall(double dx, const Vec3d& ctr, double R, int lo, int hi)
{
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3 * dx));
    Int32Grid::Ptr lab = Int32Grid::create(0);
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
                    la.setValue(c, label);
                } else {
                    pa.touchLeaf(c);
                    la.setValueOff(c, label);
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
    // Regression guard only (ruling R13): the spec 5.1 formula's curvature
    // bias measures 0.0262*dx here, and the exact field scores 0.0109*dx on
    // this metric. The spec accuracy gates are tested in Task 3.
    EXPECT_LT(sumErr / double(nPairs), 0.03);
}

TEST_F(TestLevelSetMultiObject, RedistanceOrderIndependent)
{
    // Three labels over several leaves: a ball split into labels 1 | 2 by the
    // plane x = 11.5, label 3 outside; phi starts as a rough guess. Inactive
    // voxels of [-1, 24]^3 hold their deep-interior label (ruling R7b).
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
        for (int i = -1; i <= 24; ++i)
            for (int j = -1; j <= 24; ++j)
                for (int k = -1; k <= 24; ++k) {
                    const double r = (Vec3d(i, j, k) - ctr).length();
                    const Coord c(i, j, k);
                    const int label = r < R ? (i < 12 ? 1 : 2) : 3;
                    const bool inside = i >= 0 && i < 24 && j >= 0 && j < 24 && k >= 0 && k < 24;
                    if (inside && std::abs(r - R) <= 4.0) {
                        pa.setValue(c, float(std::abs(r - R) * dx * (1.0 + 0.1 * ((i * 7 + j * 3 + k) % 5))));
                        la.setValue(c, label);
                    } else {
                        pa.touchLeaf(c);
                        la.setValueOff(c, label);
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

    // A seed whose only other-side neighbour is inactive has no usable
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

// Interface layer: active voxels with an active face neighbour of opposite
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

// Ruling R11: every seed (an active voxel with a face neighbour, active or
// not, of a different label, as the redistancer sees it) lies within one
// voxel of the true interface; distToInterface takes world coordinates.
void
expectNoSpuriousSeeds(const MLS& m, std::function<double(Vec3d)> distToInterface)
{
    const double dx = m.voxelSize();
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    tree::ValueAccessor<const Int32Tree> lab(m.labels()->tree());
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

// Spheres of radius 10*dx at (-+15*dx, 0, 0), labels 1 and 2, via
// fromSignedGrids.
MLS::Ptr
makeTwoSpheres(double dx)
{
    const double r = 10.0 * dx, cx = 15.0 * dx;
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(float(-cx), 0, 0), float(dx));
    in[2] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(float(cx), 0, 0), float(dx));
    return MLS::fromSignedGrids(in);
}

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

    // Index-space distance from a voxel centre to the face planes of a box
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
    // (fromLabelGrid redistances without reseeding, ruling R16).
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

// Three labels meeting at 120 degrees along a straight junction line through
// index point c with unit direction a. In the plane frame (u, v) normal to a
// (u = x projected off a, v = a x u, so u = x and v = y for a z-parallel
// line), label l (1..3) owns the polar angles
// [theta0 + (l-1)*120deg, theta0 + l*120deg) about the line.
struct TripleJunction
{
    double dx = 0.5;
    double theta0;
    Vec3d c, a, u, v;  // index space
    int H;             // active box [-H, H-1]^3

    TripleJunction(double th, const Vec3d& centre, const Vec3d& dir, int h = 12)
        : theta0(th), c(centre), a(dir.unitSafe()), H(h)
    {
        const Vec3d ref = std::abs(a[0]) < 0.9 ? Vec3d(1, 0, 0) : Vec3d(0, 1, 0);
        u = (ref - a * ref.dot(a)).unitSafe();
        v = a.cross(u);
    }

    Vec2d local(const Vec3d& w) const
    {
        const Vec3d d = w / dx - c;
        return Vec2d(d.dot(u), d.dot(v));
    }
    Vec2d ray(int k) const
    {
        const double t = theta0 + k * 2.0 * M_PI / 3.0;
        return Vec2d(std::cos(t), std::sin(t));
    }
    int sector(const Vec3d& w) const
    {
        const Vec2d p = local(w);
        double t = std::atan2(p[1], p[0]) - theta0;
        while (t < 0.0) t += 2.0 * M_PI;
        while (t >= 2.0 * M_PI) t -= 2.0 * M_PI;
        return std::min(3, int(t / (2.0 * M_PI / 3.0)) + 1);
    }
    // Distance from p to the ray {t*r, t >= 0} (index units).
    static double rayDist(const Vec2d& p, const Vec2d& r)
    {
        const double t = p.dot(r);
        return t >= 0.0 ? (p - r * t).length() : p.length();
    }
    // Analytic signed distance to sector @a l, world units.
    double signedDist(const Vec3d& w, int l) const
    {
        const Vec2d p = local(w);
        const double d = std::min(rayDist(p, ray(l - 1)), rayDist(p, ray(l)));
        return dx * (sector(w) == l ? -d : d);
    }
    double lineDist(const Vec3d& w) const { return local(w).length(); } // index units
    // Chebyshev distance (index units) from the voxel centre to the box faces.
    double boxDepth(const Coord& p) const
    {
        return H - std::max({std::abs(p[0] + 0.5), std::abs(p[1] + 0.5), std::abs(p[2] + 0.5)});
    }

    // The pair built directly from analytic data: phi = distance to the
    // voxel's own sector boundary, active where phi < 3*dx inside the box,
    // and a one-voxel inactive shell holding the labels (ruling R7b).
    MLS::Ptr build() const
    {
        const int halfWidth = 3;
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
        Int32Grid::Ptr lab = Int32Grid::create(0);
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
                        la.setValue(q, l);
                    } else {
                        pa.touchLeaf(q);
                        la.setValueOff(q, l);
                    }
                }
        return MLS::Ptr(new MLS(phi, lab, halfWidth));
    }
};

// Seeded junction set (ruling R19): the original fixture (theta0 = 10deg,
// z-parallel through (0.37, -0.21)), seven z-parallel lines with random
// theta0 and sub-voxel offset, and four tilted lines. Uses its own
// generator arithmetic so the set is the same on every platform.
std::vector<TripleJunction>
junctionSet()
{
    std::mt19937 rng(20261108u);
    auto u01 = [&rng]() { return double(rng()) / 4294967296.0; };
    std::vector<TripleJunction> set;
    set.emplace_back(10.0 * M_PI / 180.0, Vec3d(0.37, -0.21, 0.0), Vec3d(0, 0, 1));
    for (int n = 0; n < 7; ++n) {
        const double th = u01() * 2.0 * M_PI;
        const Vec3d c(u01() - 0.5, u01() - 0.5, u01() - 0.5);
        set.emplace_back(th, c, Vec3d(0, 0, 1));
    }
    for (int n = 0; n < 4; ++n) {
        const double th = u01() * 2.0 * M_PI;
        const Vec3d c(u01() - 0.5, u01() - 0.5, u01() - 0.5);
        const Vec3d dir(u01() - 0.5, u01() - 0.5, 1.0);
        set.emplace_back(th, c, dir);
    }
    return set;
}

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
            // Gates (ruling R19; psi_l is not smooth at the junction line):
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

            // Regression guard (spec 5.3): raw +-phi underestimates psi_l
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
            if (std::abs(analytic(w)) > 0.5 * dx) EXPECT_EQ(*it < 0.0f, analytic(w) < 0.0);
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
// LabelAwareStencil and MultiObjectForces (Task 5)
// ---------------------------------------------------------------------------

namespace {

using Stencil = tools::LabelAwareStencil<MLS>;
using Forces = tools::MultiObjectForces<MLS>;

// First active voxel of label 1 whose +x neighbour is label 2 (interface).
Coord
sphereInterfaceVoxel(const MLS& m)
{
    // Sphere 1 is centred at (-15,0,0) dx with radius 10 dx: its surface
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
    // sphere (voxels of label 1 with a face neighbour of another label).
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
    const Coord ijk(15, 0, 0); // centre of sphere 2
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
    // 3^3 neighbourhood, all label 1, so psi = -phi everywhere.
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    Int32Grid::Ptr lab = Int32Grid::create(0);
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    float v[3][3][3]; // v[i+1][j+1][k+1] = psi
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k) {
                const float d = 0.8f + 0.35f * i - 0.2f * j + 0.1f * k + 0.07f * i * i
                    + 0.05f * j * j - 0.03f * k * k + 0.04f * i * j - 0.02f * i * k + 0.06f * j * k;
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
    f.setCurvatureWeight(wc);
    f.setAdvection(vg, wa);
    f.setPressure(pg, wp); // NaN target: values used as given

    Stencil s(mls);
    s.moveTo(Coord(0, 0, 0), 1);
    const float got = f(Coord(0, 0, 0), 1, s);

    // Wedge pressureAndAdvectionMotion, evaluated by hand (dx = 1).
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
    // R26: interface curvature k_x/(1 - psi*k_x) (deviation from wedge)
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
    f2.setPressure(pg, wp);
    const float got2 = f2(Coord(0, 0, 0), 1, s);
    EXPECT_NEAR(got2, 1.4f * std::sqrt(gsNeg), 1e-6f);
}

TEST_F(TestLevelSetMultiObject, ForcesFixedAndInactive)
{
    Forces f;
    // Label 0 is passive in the evolver (R3/R20), not fixed in the functor.
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
// Evolver (spec section 4, tests 4-7)

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
// 26 neighbours.
Index64
countOrphans(const MLS& m)
{
    tree::ValueAccessor<const Int32Tree> lab(m.labels()->tree());
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

// Spec section 8 test 4: the interiors {psi_l < 0} of the per-label
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
    tree::ValueAccessor<const Int32Tree> lab(m.labels()->tree());
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
    tools::LevelSetMultiObject<MLS, RandomSpeed> lsmo(*m, speed);
    for (int n = 0; n < 50; ++n) {
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

TEST_F(TestLevelSetMultiObject, PressureGrowth)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    ConstantSpeed speed;
    speed.s = -1.0f; // dpsi/dt = -1: the object grows at unit speed
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    const size_t steps = lsmo.advect(0.0, 4.0);
    EXPECT_GE(steps, size_t(4));
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    std::cout << "PressureGrowth: steps " << steps << " mean radius " << r << std::endl;
    EXPECT_NEAR(r, 12.0, 0.5 * dx);
}

TEST_F(TestLevelSetMultiObject, CurvatureFlow)
{
    auto m = makeSphere(10.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    const size_t steps = lsmo.advect(0.0, 20.0);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    std::cout << "CurvatureFlow: steps " << steps << " r^2 " << r * r << " (expected 60)" << std::endl;
    EXPECT_NEAR(r * r, 60.0, 2.0 * std::sqrt(60.0) * 0.5 * dx);
}

TEST_F(TestLevelSetMultiObject, UniformAdvection)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    const Vec3d c0 = centroid(zeroCrossings(*m->extractSignedDistance(1)));
    const size_t steps = lsmo.advect(0.0, 5.0);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const auto pts = zeroCrossings(*m->extractSignedDistance(1));
    const Vec3d c1 = centroid(pts);
    std::cout << "UniformAdvection: steps " << steps << " shift " << (c1 - c0)
              << " mean radius " << meanRadius(pts, c1) << std::endl;
    EXPECT_NEAR(c1[0] - c0[0], 5.0, 0.5 * dx);
    EXPECT_NEAR(c1[1] - c0[1], 0.0, 0.5 * dx);
    EXPECT_NEAR(c1[2] - c0[2], 0.0, 0.5 * dx);
}

TEST_F(TestLevelSetMultiObject, FixedLabels)
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
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    for (int n = 0; n < 20; ++n) {
        lsmo.step(0.75f); // sphere 1 grows ~15 voxels: it reaches the wall
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

TEST_F(TestLevelSetMultiObject, BandMaintenance)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    ConstantSpeed speed;
    speed.s = -1.0f;
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    double t = 0.0;
    int n = 0;
    while (t < 4.0) {
        t += lsmo.step(float(4.0 - t));
        ++n;
        ASSERT_TRUE(m->phi()->tree().hasSameTopology(m->labels()->tree())) << "step " << n;
        ASSERT_LE(maxActivePhi(*m), float((m->halfWidth() + 1) * dx)) << "step " << n;
        // inactive phi is the background (R7b)
        for (auto it = m->phi()->tree().cbeginValueOff(); it; ++it) {
            ASSERT_EQ(*it, m->phi()->background()) << "step " << n;
        }
    }
    EXPECT_GE(n, 4);
}

TEST_F(TestLevelSetMultiObject, EmptyStep)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    Int32Grid::Ptr lab = Int32Grid::create(0);
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
    const Int32Tree labBefore(m.labels()->tree());
    ConstantSpeed speed;
    speed.s = -1.0f;
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(m, speed);
    EXPECT_EQ(lsmo.step(0.5f), 0.5f);
    EXPECT_TRUE(m.phi()->tree().hasSameTopology(phiBefore));
    EXPECT_TRUE(m.labels()->tree().hasSameTopology(labBefore));
    for (auto it = phiBefore.cbeginValueOn(); it; ++it) {
        EXPECT_EQ(m.phi()->tree().getValue(it.getCoord()), *it);
        EXPECT_EQ(m.labels()->tree().getValue(it.getCoord()), 0);
    }
}

TEST_F(TestLevelSetMultiObject, InterruptMidAdvect)
{
    auto m = makeSphere(8.0);
    ConstantSpeed speed;
    speed.s = -1.0f;
    CountingInterrupter interrupter;
    tools::LevelSetMultiObject<MLS, ConstantSpeed, CountingInterrupter> lsmo(*m, speed, &interrupter);
    // uninterrupted, t = 10 needs at least 10 steps (dt <= cfl*dx)
    const size_t steps = lsmo.advect(0.0, 10.0);
    EXPECT_LT(steps, size_t(10));
    EXPECT_EQ(steps, size_t(2));
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
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    for (int n = 0; n < 30; ++n) {
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
    tools::LevelSetMultiObject<MLS, RandomSpeed> serial(*a, speed), parallel(*b, speed);
    serial.setGrainSize(0);
    for (int n = 0; n < 10; ++n) {
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

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, ContactNoCascade)
{
    // Review probe: label 1 slab x < 10, label 2 block x in [10,13), y < 3,
    // in contact; label 2 is faster. Along the contact shell (10, y, 12)
    // label 2 may advance at most cfl*dx + 0.5*dx per step (ruling R23).
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
    tools::LevelSetMultiObject<MLS, LabelSpeed> lsmo(*m, speed);
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
    double t = 0.0;
    for (int s = 1; s <= 3; ++s) {
        const float dt = lsmo.step(1.0f);
        EXPECT_FLOAT_EQ(dt, float(0.5 * cfl)); // max|F| = 1: half a voxel (R24)
        t += dt;
        counts.push_back(count());
        extents.push_back(extent());
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
    }
    std::cout << "ContactNoCascade: extent " << extents[0] << " " << extents[1] << " "
              << extents[2] << " " << extents[3] << ", label-2 voxels " << counts[0] << " "
              << counts[1] << " " << counts[2] << " " << counts[3] << std::endl;
    for (int s = 1; s <= 3; ++s) {
        EXPECT_LE(extents[s] - extents[s - 1], int(std::floor(cfl + 0.5))) << "step " << s;
    }
    // cumulative: the front (speed 1) has moved t; one voxel of slack at the
    // junction with label 1, where psi_2 is only the pair lower bound
    EXPECT_LE(extents[3] - extents[0], int(std::floor(t + 1.0 + 1e-6)));
    // No cascade anywhere: after 3 steps every label-2 voxel (active or not)
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
    tools::LevelSetMultiObject<MLS, LabelSpeed> lsmo(*m, speed);
    const Coord wallSide(7, 8, 8), retreat(8, 8, 8);
    const float wallPhi = m->phi()->tree().getValue(wallSide);
    ASSERT_EQ(m->labels()->tree().getValue(retreat), 1);
    // two steps of dt = 0.5*cfl (R24) move the label-1 boundary ~1 voxel
    lsmo.step(1.0f);
    lsmo.step(1.0f);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labels()->tree().getValue(retreat), 0);
    EXPECT_EQ(m->labels()->tree().getValue(wallSide), 2);
    EXPECT_EQ(m->phi()->tree().getValue(wallSide), wallPhi);
}


////////////////////////////////////////
// Spec section 8, test 8: single-object equivalence with OpenVDB's single
// level set path (LevelSetAdvection / LevelSetFilter, FIRST_BIAS + TVD_RK1).

namespace {

// Dense Vec3SGrid over the index box @a box (voxel size @a dx), each voxel
// set to f(its world position); background zero.
Vec3SGrid::Ptr
makeVelocity(const CoordBBox& box, double dx, const std::function<Vec3s(const Vec3d&)>& f)
{
    Vec3SGrid::Ptr vel = Vec3SGrid::create(Vec3s(0.0f));
    vel->setTransform(math::Transform::createLinearTransform(dx));
    vel->denseFill(box, Vec3s(0.0f), /*active=*/true);
    tree::LeafManager<Vec3STree> leaves(vel->tree());
    leaves.foreach([&](Vec3STree::LeafNodeType& leaf, size_t) {
        for (auto it = leaf.beginValueOn(); it; ++it) {
            it.setValue(f(vel->indexToWorld(it.getCoord())));
        }
    });
    return vel;
}

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

// Fixture: sphere r = 12 dx, centred off the rotation axis so the rotational
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
referenceAdvect(FloatGrid& ref, const Vec3SGrid& vel, double t0, double t1)
{
    using Field = tools::DiscreteField<Vec3SGrid>;
    Field field(vel);
    tools::LevelSetAdvection<FloatGrid, Field> adv(ref, field);
    adv.setSpatialScheme(math::FIRST_BIAS);
    adv.setTemporalScheme(math::TVD_RK1);
    adv.advect(t0, t1);
}

void
referenceCurvature(FloatGrid& ref)
{
    tools::LevelSetFilter<FloatGrid> filter(ref);
    filter.meanCurvature(); // dt = dx^2/3
}

void
checkEquivalence(const char* name, const MLS& m, const FloatGrid& ref, size_t steps,
    double angle, const FloatGrid* analytic = nullptr, double exactRadius = -1.0)
{
    std::string msg;
    ASSERT_TRUE(m.validate(&msg)) << msg;
    const double dx = m.voxelSize();
    FloatGrid::Ptr sdf = m.extractSignedDistance(1);
    // the centre, rotated by @a angle about z, is inside
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
        // Ruling R28: mean zero-crossing radius about the exact centre @a c;
        // MOGAC's radius error at most max(1.1 x the reference's, 0.25*dx).
        const double rm = meanRadius(zeroCrossings(*sdf), c);
        const double rr = meanRadius(zeroCrossings(ref), c);
        const double em = std::abs(rm - exactRadius), er = std::abs(rr - exactRadius);
        std::cout << name << ": exact radius " << exactRadius << ", MOGAC mean radius " << rm
                  << " (error " << em / dx << " dx), reference mean radius " << rr
                  << " (error " << er / dx << " dx)" << std::endl;
        EXPECT_LE(em, std::max(1.1 * er, 0.25 * dx));
        return;
    }
    if (!analytic) {
        EXPECT_LT(d.symDiff, 0.01 * vol);
        return;
    }
    // Ruling R25: both first-order schemes diffuse, by different amounts, so
    // MOGAC must be at least as close to the exact solution as the reference
    // (10% slack) instead of within 1% of the reference.
    const ZeroSetDiff dm = compareZeroSets(*sdf, *analytic), dr = compareZeroSets(ref, *analytic);
    std::cout << name << ": vs analytic: MOGAC symmetric difference " << dm.symDiff
              << " Hausdorff " << dm.hausdorff / dx << " dx; reference " << dr.symDiff
              << " Hausdorff " << dr.hausdorff / dx << " dx" << std::endl;
    EXPECT_LE(dm.symDiff, 1.1 * dr.symDiff);
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, SingleObjectAdvection)
{
    Vec3SGrid::Ptr vel = makeRotation();
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    referenceAdvect(*ref, *vel, 0.0, 20.0);

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setAdvection(vel, 1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    const size_t steps = lsmo.advect(0.0, 20.0);
    // exact solution: the sphere rotated by 0.05*20 = 1 radian about z
    const double angle = 0.05 * 20.0;
    FloatGrid::Ptr exact = tools::createLevelSetSphere<FloatGrid>(float(kEqRadius),
        Vec3f(float(kEqCenter[0] * std::cos(angle)), float(kEqCenter[0] * std::sin(angle)), 0.0f),
        1.0f);
    checkEquivalence("SingleObjectAdvection", *m, *ref, steps, angle, exact.get());
}

TEST_F(TestLevelSetMultiObject, SingleObjectCurvature)
{
    // 60 iterations of dt = dx^2/3: total t = 20
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    for (int n = 0; n < 60; ++n) referenceCurvature(*ref);

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    const size_t steps = lsmo.advect(0.0, 20.0);
    // exact: r^2 = r0^2 - 2t at the unmoved centre
    checkEquivalence("SingleObjectCurvature", *m, *ref, steps, 0.0, nullptr,
        std::sqrt(kEqRadius * kEqRadius - 2.0 * 20.0));
}

TEST_F(TestLevelSetMultiObject, SingleObjectCombined)
{
    // Reference: 30 x (advection over dx^2/3, one meanCurvature iteration),
    // total t = 10.
    Vec3SGrid::Ptr vel = makeRotation();
    FloatGrid::Ptr ref = tools::createLevelSetSphere<FloatGrid>(
        float(kEqRadius), Vec3f(kEqCenter), 1.0f);
    const double h = 1.0 / 3.0;
    for (int n = 0; n < 30; ++n) {
        referenceAdvect(*ref, *vel, n * h, (n + 1) * h);
        referenceCurvature(*ref);
    }

    auto m = makeSphere(kEqRadius, kEqCenter);
    Forces forces;
    forces.setAdvection(vel, 1.0f);
    forces.setCurvatureWeight(1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    const size_t steps = lsmo.advect(0.0, 10.0);
    // exact: r^2 = r0^2 - 2t about the centre rotated by 0.05*10 radians
    checkEquivalence("SingleObjectCombined", *m, *ref, steps, 0.05 * 10.0, nullptr,
        std::sqrt(kEqRadius * kEqRadius - 2.0 * 10.0));
}


////////////////////////////////////////
// Spec section 8, test 9: eight objects in a tornado vortex.

namespace {

// Voxel count per label over the whole label tree (active and inactive
// voxels, tiles counted by their voxel count; label 0 excluded).
std::map<int32_t, Index64>
labelVoxelCounts(const MLS& m)
{
    std::map<int32_t, Index64> n;
    for (auto it = m.labels()->tree().cbeginValueAll(); it; ++it) {
        if (*it == 0) continue;
        n[*it] += it.getVoxelCount();
    }
    return n;
}

// Distinct unordered label pairs (a, b), a != b, both non-zero, with
// face-adjacent voxels (one of them active).
std::set<std::pair<int32_t, int32_t>>
contactPairs(const MLS& m)
{
    std::set<std::pair<int32_t, int32_t>> pairs;
    tree::ValueAccessor<const Int32Tree> lab(m.labels()->tree());
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    for (auto it = m.labels()->tree().cbeginValueOn(); it; ++it) {
        const int32_t a = *it;
        if (a == 0) continue;
        for (int f = 0; f < 6; ++f) {
            const int32_t b = lab.getValue(it.getCoord() + offs[f]);
            if (b != 0 && b != a) pairs.insert({std::min(a, b), std::max(a, b)});
        }
    }
    return pairs;
}

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

// Tornado field (ruling R27):
// - Rankine-style swirl about z: u_theta = swirl*(r/R) for r < R,
//   swirl*(R/r) outside (divergence-free);
// - radial inflow towards the ring r = rs: u_r = -inflow*tanh((r - rs)/w),
//   i.e. translation at ~inflow away from the ring and a convergence zone of
//   width ~w around it (compressive only inside that zone);
// - updraft u_z = updraft*exp(-r^2/(2 R^2)) (divergence-free).
// An exact flow never brings disjoint objects into contact, and an
// axisymmetric flow keeps the angular gaps of objects at one radius, so
// contact comes from objects on the same ray pressed together in the
// convergence zone until their gap is below a voxel.
struct Tornado
{
    double R = 16.0, swirl = 0.1, inflow = 0.2, rs = 38.5, w = 1.0, updraft = 0.2,
        z0 = -20.0;
    Vec3s operator()(const Vec3d& p) const
    {
        const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        const double ut = swirl * (r < R ? r / R : R / r);
        const double ur = -inflow * std::tanh((r - rs) / w);
        const double uz = updraft * std::exp(-r * r / (2.0 * R * R));
        Vec3d v(0.0, 0.0, uz);
        if (r > 0.0) v += Vec3d(-p[1], p[0], 0.0) * (ut / r) + Vec3d(p[0], p[1], 0.0) * (ur / r);
        return Vec3s(v);
    }
};

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, TornadoEightObjects)
{
    // Parameters (ruling R27; the brief's first set dissolved every object
    // through first-order upwind diffusion before any contact, and a ring of
    // objects at one radius cannot touch in an axisymmetric flow):
    // - eight shapes, labels 1-8, circumradius 10 (all >= 10 voxels across):
    //   sphere r = 10; cube side 12; tetrahedron, octahedron, dodecahedron,
    //   icosahedron with factory scale (circumradius) 10; capsule along z with
    //   radius 5 and axis endpoints +-5; closed ring = 8-segment polygon of
    //   radius 7 in the xy plane with tube radius 3. No rotations.
    // - four rays at angles k*90 degrees in the z = -20 plane; label k+1
    //   inside and label k+5 outside the ring r = rs = 38.5 (between voxel
    //   centres) on ray k, each placed so its surface point nearest the ring
    //   is 1.5 voxels from it (facing gap 3 voxels); dx = 1, halfWidth 3.
    // - Tornado field (see Tornado): R = 16, swirl 0.1, inflow 0.2 towards
    //   rs = 38.5 with zone width 1, updraft 0.2; dense over [-64, 64]^3;
    //   advection weight 1, curvature weight 0.01.
    // - Advect in chunks of 2 until >= 4 contact pairs, capped at t = 40.
    // Volume gate: each label's voxel loss <= 1.1 x the loss of the same
    // shape advected alone by LevelSetAdvection (FIRST_BIAS / TVD_RK1) in the
    // same field over the same time, + 2% absolute.
    const float dx = 1.0f, hw = 3.0f;
    const Tornado tornado;
    auto make = [&](int32_t l, const Vec3f& c) -> FloatGrid::Ptr {
        switch (l) {
        case 1: return tools::createLevelSetSphere<FloatGrid>(10.0f, c, dx, hw);
        case 2: return tools::createLevelSetCube<FloatGrid>(12.0f, c, dx, hw);
        case 3: return tools::createLevelSetTetrahedron<FloatGrid>(10.0f, c, dx, hw);
        case 4: return tools::createLevelSetOctahedron<FloatGrid>(10.0f, c, dx, hw);
        case 5: return tools::createLevelSetDodecahedron<FloatGrid>(10.0f, c, dx, hw);
        case 6: return tools::createLevelSetIcosahedron<FloatGrid>(10.0f, c, dx, hw);
        case 7:
            return tools::createLevelSetCapsule<FloatGrid>(
                c - Vec3f(0.0f, 0.0f, 5.0f), c + Vec3f(0.0f, 0.0f, 5.0f), 5.0f, dx, hw);
        default: {
            std::vector<Vec3f> verts;
            std::vector<Vec2I> segs;
            for (int i = 0; i < 8; ++i) {
                const double a = i * math::pi<double>() / 4.0;
                verts.push_back(c + Vec3f(float(7.0 * std::cos(a)), float(7.0 * std::sin(a)), 0.0f));
                segs.push_back(Vec2I(i, (i + 1) % 8));
            }
            return tools::createLevelSetTubeComplex<FloatGrid>(verts, segs, 3.0f, dx, hw);
        }
        }
    };
    // extent of a shape centred at the origin along unit direction d: max of
    // p.d - psi(p) over interior voxels (psi < 0)
    auto extent = [](const FloatGrid& g, const Vec3d& d) {
        double e = 0.0;
        for (auto it = g.tree().cbeginValueOn(); it; ++it) {
            if (*it < 0.0f) e = std::max(e, g.indexToWorld(it.getCoord()).dot(d) - double(*it));
        }
        return e;
    };
    std::map<int32_t, FloatGrid::ConstPtr> in;
    for (int32_t l = 1; l <= 8; ++l) {
        const double a = ((l - 1) % 4) * math::pi<double>() / 2.0;
        const Vec3d ray(std::cos(a), std::sin(a), 0.0);
        const bool inner = l <= 4;
        const double e = extent(*make(l, Vec3f(0.0f)), inner ? ray : -ray);
        const double rad = inner ? tornado.rs - 1.5 - e : tornado.rs + 1.5 + e;
        in[l] = make(l, Vec3f(ray * rad + Vec3d(0.0, 0.0, tornado.z0)));
    }
    auto m = MLS::fromSignedGrids(in, int(hw));
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
    const std::map<int32_t, Index64> vol0 = labelVoxelCounts(*m);
    ASSERT_TRUE(contactPairs(*m).empty());

    Vec3SGrid::Ptr vel = makeVelocity(CoordBBox(Coord(-64), Coord(64)), dx, tornado);
    Forces forces;
    forces.setAdvection(vel, 1.0f);
    forces.setCurvatureWeight(0.01f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);

    const double tCap = 40.0, chunk = 2.0;
    double t = 0.0;
    size_t steps = 0;
    std::set<std::pair<int32_t, int32_t>> pairs;
    const auto start = std::chrono::steady_clock::now();
    while (t < tCap - 1e-9 && pairs.size() < 4) {
        steps += lsmo.advect(t, t + chunk);
        t += chunk;
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "t " << t << ": " << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*m, overlaps, badCover);
        ASSERT_EQ(overlaps, Index64(0)) << "t " << t;
        ASSERT_EQ(badCover, Index64(0)) << "t " << t;
        ASSERT_EQ(countNaN(*m->phi()), Index64(0)) << "t " << t;
        pairs = contactPairs(*m);
        std::cout << "Tornado: t " << t << " steps " << steps << " contacts " << pairs.size();
        for (auto& lv : labelVoxelCounts(*m)) std::cout << " " << lv.first << ":" << lv.second;
        std::cout << std::endl;
    }
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Tornado: t " << t << ", " << steps << " steps, " << secs << " s, contact pairs";
    for (const auto& pr : pairs) std::cout << " " << pr.first << "-" << pr.second;
    std::cout << std::endl;
    EXPECT_GE(pairs.size(), size_t(4));

    const std::map<int32_t, Index64> vol1 = labelVoxelCounts(*m);
    for (int32_t l = 1; l <= 8; ++l) {
        // reference: the same shape alone, LevelSetAdvection to the same t
        FloatGrid::Ptr ref = in[l]->deepCopy();
        const double r0 = double(insideVoxelCount(*ref));
        referenceAdvect(*ref, *vel, 0.0, t);
        const double refLoss = 1.0 - double(insideVoxelCount(*ref)) / r0;
        const double v0 = double(vol0.at(l));
        const double v1 = vol1.count(l) ? double(vol1.at(l)) : 0.0;
        const double loss = 1.0 - v1 / v0;
        std::cout << "Tornado: label " << l << " voxels " << v0 << " -> " << v1 << " (loss "
                  << 100.0 * loss << "%), reference loss " << 100.0 * refLoss << "%" << std::endl;
        EXPECT_GE(v1, 1.0) << "label " << l;
        EXPECT_LE(loss, 1.1 * refLoss + 0.02) << "label " << l;
    }
}
