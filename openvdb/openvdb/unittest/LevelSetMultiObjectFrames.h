// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file LevelSetMultiObjectFrames.h
///
/// @brief Opt-in per-step mesh dump of the MOGAC unit-test simulations
/// (visualization only).
///
/// The tests construct their evolvers as lsmo_frames::FrameEvolver, a thin
/// subclass of tools::LevelSetMultiObject. When the environment variable
/// OPENVDB_MOGAC_FRAME_DIR is unset it forwards step() and advect() to the
/// base class unchanged. When it is set, every evolver writes
///
///     <dir>/<TestName>-<k>/frame_%05d.ply   (k = evolver index within the test)
///     <dir>/<TestName>-<k>/frames.txt       ("frame time" per line)
///
/// with frame 0 the state before the first step and one frame after every
/// step. Each PLY holds the zero-level meshes of all labels from
/// tools::multiObjectToMesh with the evolver's connectivity and phi floor,
/// so topology-preserving runs are meshed with the matching digital
/// topology. Rendered by unittest/scripts/render_mogac_tests.py.

#ifndef OPENVDB_UNITTEST_LEVELSETMULTIOBJECTFRAMES_H_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_LEVELSETMULTIOBJECTFRAMES_H_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tools/LevelSetMultiObjectMesh.h>
#include <openvdb/util/NullInterrupter.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace lsmo_frames {

/// Environment variable naming the frame output directory.
inline constexpr const char* kFrameDirEnv = "OPENVDB_MOGAC_FRAME_DIR";
/// Number of entries in the label color palette.
inline constexpr int kPaletteSize = 12;
/// Digits of the zero-padded frame number in the file name.
inline constexpr int kFrameDigits = 5;

/// The frame output directory, or nullptr when frame dumping is off.
inline const char*
frameDir()
{
    const char* d = std::getenv(kFrameDirEnv);
    return (d && *d) ? d : nullptr;
}

/// RGB of @a label: a Tableau-like palette indexed by the label value, so a
/// label keeps its color for the whole simulation.
inline const unsigned char*
labelColor(int32_t label)
{
    static const unsigned char palette[kPaletteSize][3] = {
        {78, 121, 167}, {242, 142, 43}, {225, 87, 89}, {118, 183, 178},
        {89, 161, 79}, {237, 201, 72}, {176, 122, 161}, {255, 157, 167},
        {156, 117, 95}, {23, 190, 207}, {31, 119, 180}, {188, 189, 34}};
    const int i = int(((int64_t(label) - 1) % kPaletteSize + kPaletteSize) % kPaletteSize);
    return palette[i];
}

/// Directory name of the next evolver of the running test: "<Test>-<k>".
inline std::string
nextSimulationName()
{
    static std::unordered_map<std::string, int> counts;
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test = info ? info->name() : "NoTest";
    return test + "-" + std::to_string(++counts[test]);
}

/// Writes one binary little-endian PLY: float xyz, uchar rgb, int label per
/// vertex, triangle faces.
inline bool
writePly(const std::filesystem::path& path,
    const std::map<int32_t, openvdb::tools::LabelMesh>& meshes)
{
    size_t nv = 0, nf = 0;
    for (const auto& lm : meshes) {
        nv += lm.second.points.size();
        nf += lm.second.triangles.size();
    }
    FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "ply\nformat binary_little_endian 1.0\nelement vertex %zu\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int label\nelement face %zu\n"
        "property list uchar int vertex_indices\nend_header\n", nv, nf);
    for (const auto& lm : meshes) {
        const unsigned char* c = labelColor(lm.first);
        for (const openvdb::Vec3s& p : lm.second.points) {
            std::fwrite(p.asV(), sizeof(float), 3, f);
            std::fwrite(c, 1, 3, f);
            std::fwrite(&lm.first, sizeof(int32_t), 1, f);
        }
    }
    int32_t base = 0;
    for (const auto& lm : meshes) {
        for (const openvdb::Vec3I& t : lm.second.triangles) {
            const unsigned char n = 3;
            const int32_t idx[3] = {base + int32_t(t[0]), base + int32_t(t[1]),
                base + int32_t(t[2])};
            std::fwrite(&n, 1, 1, f);
            std::fwrite(idx, sizeof(int32_t), 3, f);
        }
        base += int32_t(lm.second.points.size());
    }
    return std::fclose(f) == 0;
}

/// tools::LevelSetMultiObject that also dumps a mesh after every step when
/// frameDir() is set (see the file documentation). Calls through a reference
/// to the base class do not dump.
template<typename MultiLevelSetT, typename SpeedT,
    typename InterruptT = openvdb::util::NullInterrupter>
class FrameEvolver: public openvdb::tools::LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>
{
public:
    using Base = openvdb::tools::LevelSetMultiObject<MultiLevelSetT, SpeedT, InterruptT>;

    FrameEvolver(MultiLevelSetT& mls, const SpeedT& speed, InterruptT* interrupt = nullptr)
        : Base(mls, speed, interrupt), mMls(mls), mInterrupter(interrupt) {}
    FrameEvolver(MultiLevelSetT& mls, const SpeedT& speed,
        openvdb::math::BiasedGradientScheme scheme, InterruptT* interrupt = nullptr)
        : Base(mls, speed, scheme, interrupt), mMls(mls), mInterrupter(interrupt) {}

    float step(float maxDt)
    {
        if (!frameDir()) return Base::step(maxDt);
        if (mFrame == 0) this->dump(); // the state before the first step
        const float dt = Base::step(maxDt);
        if (dt > 0.0f) this->dump();
        return dt;
    }

    /// Same loop as the base class advect(), stepping through this->step().
    /// With an interrupter it runs the base class advect() unchanged (whose
    /// interrupter checks depend on state private to the base class) and
    /// dumps only the states before and after it.
    size_t advect(double t0, double t1)
    {
        if (!frameDir()) return Base::advect(t0, t1);
        if (mInterrupter) {
            if (mFrame == 0) {
                this->setTime(t0);
                this->dump();
            }
            const size_t n = Base::advect(t0, t1);
            if (n > 0) this->dump();
            return n;
        }
        size_t n = 0;
        double t = t0;
        const double eps = openvdb::tools::lsmo_internal::kAdvectTimeRelativeEpsilon
            * std::max(1.0, std::abs(t1));
        while (t1 - t > eps) {
            this->setTime(t);
            const float dt = this->step(float(t1 - t));
            if (!(dt > 0.0f)) break;
            t += dt;
            ++n;
        }
        return n;
    }

private:
    void dump()
    {
        if (mName.empty()) {
            mName = nextSimulationName();
            mDir = std::filesystem::path(frameDir()) / mName;
            std::filesystem::create_directories(mDir);
            std::filesystem::remove(mDir / "frames.txt");
        }
        openvdb::tools::MultiObjectMeshSettings settings;
        settings.connectivity = this->connectivity();
        settings.minimumPhi = this->phiFloor();
        const auto meshes = openvdb::tools::multiObjectToMesh(mMls, {}, settings);
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%0*d.ply", kFrameDigits, mFrame);
        EXPECT_TRUE(writePly(mDir / name, meshes)) << "cannot write " << (mDir / name);
        if (FILE* f = std::fopen((mDir / "frames.txt").string().c_str(), "a")) {
            std::fprintf(f, "%d %.9g\n", mFrame, this->time());
            std::fclose(f);
        }
        ++mFrame;
    }

    MultiLevelSetT& mMls;
    InterruptT* mInterrupter;
    std::string mName;
    std::filesystem::path mDir;
    int mFrame = 0;
};

} // namespace lsmo_frames

#endif // OPENVDB_UNITTEST_LEVELSETMULTIOBJECTFRAMES_H_HAS_BEEN_INCLUDED
