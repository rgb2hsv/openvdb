// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
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

// Three labels meeting at 120 degrees along a z-parallel junction axis
// through index (0.37, -0.21): label l (1..3) owns the polar angles
// [theta0 + (l-1)*120deg, theta0 + l*120deg) about the axis, theta0 = 10deg.
struct TripleJunction
{
    double dx = 0.5;
    double x0 = 0.37, y0 = -0.21;            // axis, index space
    double theta0 = 10.0 * M_PI / 180.0;

    Vec2d local(const Vec3d& w) const { return Vec2d(w[0] / dx - x0, w[1] / dx - y0); }
    Vec2d ray(int k) const
    {
        const double a = theta0 + k * 2.0 * M_PI / 3.0;
        return Vec2d(std::cos(a), std::sin(a));
    }
    int sector(const Vec3d& w) const
    {
        const Vec2d p = local(w);
        double a = std::atan2(p[1], p[0]) - theta0;
        while (a < 0.0) a += 2.0 * M_PI;
        while (a >= 2.0 * M_PI) a -= 2.0 * M_PI;
        return std::min(3, int(a / (2.0 * M_PI / 3.0)) + 1);
    }
    // Distance from p to the ray {t*u, t >= 0} (index units).
    static double rayDist(const Vec2d& p, const Vec2d& u)
    {
        const double t = p.dot(u);
        return t >= 0.0 ? (p - u * t).length() : p.length();
    }
    // Analytic signed distance to sector @a l, world units.
    double signedDist(const Vec3d& w, int l) const
    {
        const Vec2d p = local(w);
        const double d = std::min(rayDist(p, ray(l - 1)), rayDist(p, ray(l)));
        return dx * (sector(w) == l ? -d : d);
    }
    double axisDist(const Vec3d& w) const { return local(w).length(); } // index units

    // The pair built directly from analytic data: phi = distance to the
    // voxel's own sector boundary, active where phi < 3*dx inside
    // [-20, 19]^3, and a one-voxel inactive shell holding the labels
    // (ruling R7b).
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
        for (int i = -21; i <= 20; ++i)
            for (int j = -21; j <= 20; ++j)
                for (int k = -21; k <= 20; ++k) {
                    const Coord c(i, j, k);
                    const Vec3d w = xform->indexToWorld(c);
                    const int l = sector(w);
                    const double d = std::abs(signedDist(w, l));
                    const bool inBox = std::max({std::abs(i + 0.5), std::abs(j + 0.5),
                        std::abs(k + 0.5)}) < 20.0;
                    if (inBox && d < halfWidth * dx) {
                        pa.setValue(c, float(d));
                        la.setValue(c, l);
                    } else {
                        pa.touchLeaf(c);
                        la.setValueOff(c, l);
                    }
                }
        return MLS::Ptr(new MLS(phi, lab, halfWidth));
    }
};

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
    const TripleJunction tj;
    auto m = tj.build();
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));
    const double dx = tj.dx;
    const Coord* offs = tools::lsmo_internal::faceOffsets();

    for (int l = 1; l <= 3; ++l) {
        SCOPED_TRACE("label " + std::to_string(l));
        const auto analytic = [&tj, l](Vec3d w) { return tj.signedDist(w, l); };

        FloatGrid::Ptr g = m->extractSignedDistance(l);
        ASSERT_TRUE(g);
        expectLevelSetGrid(*g, *m, 3);

        const double maxErr = maxInterfaceError(*g, analytic);
        const double rms = rmsBandError(*g, analytic);

        // Interface-layer error binned by distance to the junction axis.
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
            while (tj.axisDist(w) >= edges[b]) ++b;
            ++binN[b];
            binMax[b] = std::max(binMax[b], std::abs(*it - analytic(w)) / dx);
        }
        std::cout << "TripleJunction label " << l << ": max interface error " << maxErr
            << " dx, RMS band error " << rms << " dx; interface max by axis distance"
            << " [<1: " << binMax[0] << " (" << binN[0] << ")"
            << ", 1-2: " << binMax[1] << " (" << binN[1] << ")"
            << ", 2-4: " << binMax[2] << " (" << binN[2] << ")"
            << ", >4: " << binMax[3] << " (" << binN[3] << ")]" << std::endl;
        // Gates (ruling R18): interface voxels farther than 1 voxel from the
        // junction axis < 0.05*dx, within 1 voxel < 0.25*dx; RMS over the
        // whole band < 0.1*dx.
        const double farMax = std::max({binMax[1], binMax[2], binMax[3]});
        std::cout << "TripleJunction label " << l << ": interface max beyond 1 voxel of the axis "
            << farMax << " dx, within 1 voxel " << binMax[0] << " dx" << std::endl;
        EXPECT_LT(farMax, 0.05);
        EXPECT_LT(binMax[0], 0.25);
        EXPECT_LT(rms, 0.1);

        // Every voxel within halfWidth of the label boundary is in the band
        // (checked 3+ voxels away from the x/y faces of the fixture box,
        // beyond which the boundary's nearest point may lie outside the
        // truncated band).
        for (auto it = g->tree().cbeginValueOn(); it; ++it) EXPECT_LT(std::abs(*it), 3.0 * dx);
        Index64 missing = 0;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            const Coord c = it.getCoord();
            if (std::max(std::abs(c[0] + 0.5), std::abs(c[1] + 0.5)) > 16.5) continue;
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
        std::cout << "TripleJunction label " << l << ": raw max underestimate outside "
            << maxUnder << " dx" << std::endl;
        EXPECT_GT(maxUnder, 0.2);
    }
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
