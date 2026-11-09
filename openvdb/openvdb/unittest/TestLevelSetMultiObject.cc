// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>

#include <gtest/gtest.h>

#include <string>

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
    auto m = makePair();
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
    EXPECT_EQ(m->phi()->activeVoxelCount(), n0);
    EXPECT_EQ(m->phi()->activeVoxelCount(), m->labels()->activeVoxelCount());
    EXPECT_TRUE(m->validate());
}
