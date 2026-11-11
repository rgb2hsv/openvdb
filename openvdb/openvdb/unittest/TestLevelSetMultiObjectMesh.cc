// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObjectMesh.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "LevelSetMultiObjectFixtures.h"
#include "LevelSetMultiObjectTopology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class TestLevelSetMultiObjectMesh: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

namespace {

using namespace openvdb;
using Conn = tools::MultiObjectConnectivity;

// Cube geometry, rebuilt here from the numbering documented in
// LevelSetMultiObjectMesh.h rather than taken from the generator:
// corner bit i <-> (x,y,z) = (i&1, (i>>1)&1, (i>>2)&1); edges 0-3 run along
// x, 4-7 along y, 8-11 along z; edge 4a+k has its two free coordinates
// (in ascending axis order) equal to (k&1, k>>1).
struct Cube
{
    std::array<std::array<int, 2>, 12> edge;   // corner ids, lower corner first
    std::array<Vec3d, 8> corner;
    std::array<Vec3d, 12> mid;                 // edge midpoints
    std::array<int, 12> faceMask;              // bit f set if edge lies on face f
    // face f = 2*axis + side (side 0: coordinate 0); corners in cyclic order
    std::array<std::array<int, 4>, 6> faceCorners;
    std::array<Vec3d, 6> faceNormal;           // outward

    Cube()
    {
        for (int i = 0; i < 8; ++i) corner[i] = Vec3d(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        for (int a = 0; a < 3; ++a) {
            const int u = a == 0 ? 1 : 0, v = a == 2 ? 1 : 2;
            for (int k = 0; k < 4; ++k) {
                const int c0 = ((k & 1) << u) | ((k >> 1) << v);
                edge[4 * a + k] = {c0, c0 | (1 << a)};
            }
            for (int s = 0; s < 2; ++s) {
                const int f = 2 * a + s, base = s << a;
                faceCorners[f] = {base, base | (1 << u), base | (1 << u) | (1 << v),
                    base | (1 << v)};
                faceNormal[f] = Vec3d(0.0);
                faceNormal[f][a] = s ? 1.0 : -1.0;
            }
        }
        for (int e = 0; e < 12; ++e) {
            mid[e] = 0.5 * (corner[edge[e][0]] + corner[edge[e][1]]);
            faceMask[e] = 0;
            for (int f = 0; f < 6; ++f) {
                int n = 0;
                for (int c : faceCorners[f]) n += (c == edge[e][0] || c == edge[e][1]);
                if (n == 2) faceMask[e] |= 1 << f;
            }
        }
    }

    int edgeBetween(int a, int b) const
    {
        for (int e = 0; e < 12; ++e)
            if ((edge[e][0] == a && edge[e][1] == b) || (edge[e][0] == b && edge[e][1] == a))
                return e;
        return -1;
    }
};

const Cube&
cube() { static const Cube c; return c; }

int fgOf(Conn c) { return c == Conn::FG26_BG6 ? 26 : c == Conn::FG18_BG6 ? 18 : 6; }
int bgOf(Conn c) { return c == Conn::FG6_BG26 ? 26 : c == Conn::FG6_BG18 ? 18 : 6; }

using lsmo_test::connectivityName;

// cube-restricted adjacency: 6 = cube edge, 18 = + face diagonal, 26 = all
bool
adjacent(int a, int b, int n)
{
    const int d = int(((a ^ b) & 1) != 0) + int(((a ^ b) & 2) != 0) + int(((a ^ b) & 4) != 0);
    return d == 1 || (d == 2 && n >= 18) || (d == 3 && n == 26);
}

// component representative of every corner of the given type under adjacency n
std::array<int, 8>
cornerComponents(int caseIdx, bool in, int n)
{
    std::array<int, 8> rep;
    for (int i = 0; i < 8; ++i) rep[i] = i;
    auto find = [&](int x) { while (rep[x] != x) x = rep[x]; return x; };
    for (int a = 0; a < 8; ++a)
        for (int b = a + 1; b < 8; ++b) {
            if (bool((caseIdx >> a) & 1) != in || bool((caseIdx >> b) & 1) != in) continue;
            if (adjacent(a, b, n)) rep[find(a)] = find(b);
        }
    for (int i = 0; i < 8; ++i) rep[i] = find(i);
    return rep;
}

// Expected boundary segments on face f for the documented face rule: an
// ambiguous face (two diagonal in-corners) joins the in-pair when fg is 18 or
// 26 (segments cut off the out-corners) and separates it when fg is 6
// (segments cut off the in-corners). Returned as sorted edge pairs.
std::set<std::pair<int, int>>
expectedFaceSegments(int caseIdx, int f, int fg)
{
    const Cube& C = cube();
    const auto& fc = C.faceCorners[f];
    std::array<bool, 4> in;
    for (int j = 0; j < 4; ++j) in[j] = (caseIdx >> fc[j]) & 1;
    std::vector<int> crossing;
    for (int j = 0; j < 4; ++j)
        if (in[j] != in[(j + 1) % 4]) crossing.push_back(C.edgeBetween(fc[j], fc[(j + 1) % 4]));
    std::set<std::pair<int, int>> segs;
    auto add = [&](int a, int b) { segs.insert({std::min(a, b), std::max(a, b)}); };
    if (crossing.size() == 2) add(crossing[0], crossing[1]);
    if (crossing.size() == 4) {
        const bool cutIn = (fg == 6);
        for (int j = 0; j < 4; ++j) {
            if (in[j] != cutIn) continue;
            add(C.edgeBetween(fc[(j + 3) % 4], fc[j]), C.edgeBetween(fc[j], fc[(j + 1) % 4]));
        }
    }
    return segs;
}

// Segment / triangle intersection (closed triangle, closed segment).
bool
segmentHitsTriangle(const Vec3d& p0, const Vec3d& p1,
    const Vec3d& a, const Vec3d& b, const Vec3d& c)
{
    const Vec3d d = p1 - p0, e1 = b - a, e2 = c - a;
    const Vec3d h = d.cross(e2);
    const double det = e1.dot(h);
    if (std::abs(det) < 1e-14) return false; // parallel; jittered grid avoids coplanar hits
    const double inv = 1.0 / det;
    const Vec3d s = p0 - a;
    const double u = inv * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;
    const Vec3d q = s.cross(e1);
    const double v = inv * d.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;
    const double t = inv * e2.dot(q);
    return t >= 0.0 && t <= 1.0;
}

// Geometric region labels of the 8 corners: flood fill on a jittered lattice
// inside the unit cube, with lattice steps blocked by the case's triangles
// (midpoint vertices). Returns -1 for a corner that cannot reach its lattice
// point without crossing the surface.
std::array<int, 8>
geometricRegions(const std::vector<std::array<int, 3>>& tris)
{
    const Cube& C = cube();
    constexpr int N = 10;
    const Vec3d jitter(0.0123456 * std::sqrt(2.0), 0.0234567 * std::sqrt(3.0),
        -0.0154321 * std::sqrt(5.0));
    auto pos = [&](int i, int j, int k) {
        return Vec3d((i + 0.5 + jitter[0]) / N, (j + 0.5 + jitter[1]) / N, (k + 0.5 + jitter[2]) / N);
    };
    auto idx = [&](int i, int j, int k) { return i + N * (j + N * k); };
    auto blocked = [&](const Vec3d& p, const Vec3d& q) {
        for (const auto& t : tris)
            if (segmentHitsTriangle(p, q, C.mid[t[0]], C.mid[t[1]], C.mid[t[2]])) return true;
        return false;
    };
    std::vector<int> rep(N * N * N);
    for (size_t i = 0; i < rep.size(); ++i) rep[i] = int(i);
    std::function<int(int)> find = [&](int x) { while (rep[x] != x) x = rep[x] = rep[rep[x]]; return x; };
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Vec3d p = pos(i, j, k);
                if (i + 1 < N && !blocked(p, pos(i + 1, j, k))) rep[find(idx(i, j, k))] = find(idx(i + 1, j, k));
                if (j + 1 < N && !blocked(p, pos(i, j + 1, k))) rep[find(idx(i, j, k))] = find(idx(i, j + 1, k));
                if (k + 1 < N && !blocked(p, pos(i, j, k + 1))) rep[find(idx(i, j, k))] = find(idx(i, j, k + 1));
            }
    std::array<int, 8> region;
    for (int c = 0; c < 8; ++c) {
        const int i = (c & 1) ? N - 1 : 0, j = (c & 2) ? N - 1 : 0, k = (c & 4) ? N - 1 : 0;
        region[c] = blocked(C.corner[c], pos(i, j, k)) ? -1 : find(idx(i, j, k));
    }
    return region;
}

// Checks one case; returns an empty string on success, else a description.
std::string
checkCase(Conn conn, int caseIdx, const tools::lsmo_internal::CaseTriangulation& ct,
    int& tubeComponents)
{
    const Cube& C = cube();
    const int fg = fgOf(conn), bg = bgOf(conn);
    std::ostringstream err;
    auto isIn = [&](int c) { return bool((caseIdx >> c) & 1); };
    auto crosses = [&](int e) { return isIn(C.edge[e][0]) != isIn(C.edge[e][1]); };
    // in-corner and out-corner of a crossing edge
    auto inC = [&](int e) { return isIn(C.edge[e][0]) ? C.edge[e][0] : C.edge[e][1]; };
    auto outC = [&](int e) { return isIn(C.edge[e][0]) ? C.edge[e][1] : C.edge[e][0]; };

    if (ct.count > 16) { err << "count " << int(ct.count) << " > 16"; return err.str(); }
    std::vector<std::array<int, 3>> tris;
    for (int t = 0; t < ct.count; ++t)
        tris.push_back({int(ct.tris[t][0]), int(ct.tris[t][1]), int(ct.tris[t][2])});

    // 1. edge usage
    std::set<int> used;
    for (const auto& t : tris)
        for (int v : t) {
            if (v < 0 || v > 11) { err << "edge id " << v << " out of range"; return err.str(); }
            used.insert(v);
        }
    for (const auto& t : tris)
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
            err << "degenerate triangle"; return err.str();
        }
    for (int e = 0; e < 12; ++e)
        if (crosses(e) != bool(used.count(e))) {
            err << "edge " << e << (crosses(e) ? " crossing but unused" : " used but not crossing");
            return err.str();
        }

    // 2. manifoldness: directed edge multiplicities
    std::map<std::pair<int, int>, int> directed;
    for (const auto& t : tris)
        for (int j = 0; j < 3; ++j) ++directed[{t[j], t[(j + 1) % 3]}];
    std::vector<std::pair<int, int>> boundary; // directed boundary segments
    for (const auto& kv : directed) {
        const int a = kv.first.first, b = kv.first.second;
        if (kv.second != 1) { err << "directed edge " << a << "->" << b << " repeated"; return err.str(); }
        const bool reverse = directed.count({b, a}) > 0;
        const bool sameFace = (C.faceMask[a] & C.faceMask[b]) != 0;
        if (reverse) {
            if (sameFace) { err << "interior edge " << a << "-" << b << " lies in a cube face"; return err.str(); }
        } else {
            if (!sameFace) { err << "boundary edge " << a << "-" << b << " not on a cube face"; return err.str(); }
            boundary.emplace_back(a, b);
        }
    }
    // vertex links: every vertex has one outgoing and one incoming boundary
    // segment and its triangles form a single fan between them
    for (int v : used) {
        int outB = -1, inB = -1, nOut = 0, nIn = 0;
        for (const auto& s : boundary) {
            if (s.first == v) { outB = s.second; ++nOut; }
            if (s.second == v) { inB = s.first; ++nIn; }
        }
        if (nOut != 1 || nIn != 1) { err << "vertex " << v << " boundary degree"; return err.str(); }
        // walk the fan: start with the triangle holding v->outB, rotate about v
        int cur = outB, steps = 0, nTris = 0;
        for (const auto& t : tris) nTris += (t[0] == v || t[1] == v || t[2] == v);
        while (true) {
            // find triangle containing directed v->cur, its third vertex w; next edge is w->v... rotate
            int w = -1;
            for (const auto& t : tris)
                for (int j = 0; j < 3; ++j)
                    if (t[j] == v && t[(j + 1) % 3] == cur) w = t[(j + 2) % 3];
            if (w < 0) { err << "vertex " << v << " fan broken"; return err.str(); }
            ++steps;
            if (w == inB) break;
            cur = w; // triangle with v->w exists iff interior edge (w->v used above)
            if (steps > 16) break;
        }
        if (steps != nTris) { err << "vertex " << v << " link is not a single disk"; return err.str(); }
    }

    // 3. face consistency against the documented face rule
    for (int f = 0; f < 6; ++f) {
        std::set<std::pair<int, int>> got;
        for (const auto& s : boundary)
            if (C.faceMask[s.first] & C.faceMask[s.second] & (1 << f))
                got.insert({std::min(s.first, s.second), std::max(s.first, s.second)});
        if (got != expectedFaceSegments(caseIdx, f, fg)) {
            err << "face " << f << " segments differ from face rule"; return err.str();
        }
    }

    // 4. winding. (a) at the cube boundary, n_face x (b - a) points from the
    // in-corner to the out-corner of both endpoint edges;
    for (const auto& s : boundary) {
        const int f = [&] {
            for (int g = 0; g < 6; ++g) {
                if (C.faceMask[s.first] & C.faceMask[s.second] & (1 << g)) return g;
            }
            return -1;
        }();
        const Vec3d side = C.faceNormal[f].cross(C.mid[s.second] - C.mid[s.first]);
        for (int e : {s.first, s.second})
            if (side.dot(C.corner[outC(e)] - C.corner[inC(e)]) <= 0.0) {
                err << "boundary segment " << s.first << "->" << s.second << " wound inward";
                return err.str();
            }
    }
    // (b) per connected component of triangles, each triangle normal agrees
    // with (out-centroid - in-centroid) of the component's corners, or for a
    // tube around a body diagonal, points away from (in-tube) / towards
    // (out-tube) the diagonal axis.
    {
        std::vector<int> comp(tris.size());
        for (size_t i = 0; i < tris.size(); ++i) comp[i] = int(i);
        std::function<int(int)> find = [&](int x) { while (comp[x] != x) x = comp[x]; return x; };
        for (size_t i = 0; i < tris.size(); ++i)
            for (size_t j = i + 1; j < tris.size(); ++j) {
                int shared = 0;
                for (int a : tris[i]) for (int b : tris[j]) shared += (a == b);
                if (shared >= 2) comp[find(int(i))] = find(int(j));
            }
        std::map<int, std::vector<int>> groups;
        for (size_t i = 0; i < tris.size(); ++i) groups[find(int(i))].push_back(int(i));
        for (const auto& g : groups) {
            std::set<int> ins, outs;
            for (int ti : g.second) for (int e : tris[ti]) { ins.insert(inC(e)); outs.insert(outC(e)); }
            Vec3d ci(0.0), co(0.0);
            for (int c : ins) ci += C.corner[c];
            for (int c : outs) co += C.corner[c];
            ci *= 1.0 / double(ins.size());
            co *= 1.0 / double(outs.size());
            const Vec3d dir = co - ci;
            const bool tubeIn = ins.size() == 2 && (*ins.begin() ^ *ins.rbegin()) == 7;
            const bool tubeOut = outs.size() == 2 && (*outs.begin() ^ *outs.rbegin()) == 7;
            if (tubeIn || tubeOut) ++tubeComponents;
            for (int ti : g.second) {
                const auto& t = tris[ti];
                const Vec3d n = (C.mid[t[1]] - C.mid[t[0]]).cross(C.mid[t[2]] - C.mid[t[0]]);
                double d;
                if (tubeIn || tubeOut) {
                    const auto& axisSet = tubeIn ? ins : outs;
                    const Vec3d a0 = C.corner[*axisSet.begin()], a1 = C.corner[*axisSet.rbegin()];
                    const Vec3d centroid = (C.mid[t[0]] + C.mid[t[1]] + C.mid[t[2]]) / 3.0;
                    const Vec3d axis = (a1 - a0).unit();
                    const Vec3d radial = (centroid - a0) - axis * axis.dot(centroid - a0);
                    d = n.dot(radial) * (tubeIn ? 1.0 : -1.0);
                } else {
                    d = n.dot(dir);
                }
                if (!(d > 0.0)) { err << "triangle " << ti << " wound inward"; return err.str(); }
            }
        }
    }

    // 5. components: corners sharing a geometric region of the cube minus the
    // surface are exactly the in-components under fg and out-components under bg
    {
        const auto region = geometricRegions(tris);
        for (int c = 0; c < 8; ++c)
            if (region[c] < 0) { err << "corner " << c << " cut off from its lattice point"; return err.str(); }
        const auto inComp = cornerComponents(caseIdx, true, fg);
        const auto outComp = cornerComponents(caseIdx, false, bg);
        for (int a = 0; a < 8; ++a)
            for (int b = a + 1; b < 8; ++b) {
                const bool sameRegion = region[a] == region[b];
                bool expected = false;
                if (isIn(a) == isIn(b)) expected = isIn(a) ? inComp[a] == inComp[b] : outComp[a] == outComp[b];
                if (sameRegion != expected) {
                    err << "corners " << a << "," << b << (sameRegion ? " joined" : " separated")
                        << " but expected " << (expected ? "joined" : "separated");
                    return err.str();
                }
            }
    }
    return std::string();
}

} // namespace


TEST_F(TestLevelSetMultiObjectMesh, CaseTableAll)
{
    namespace li = tools::lsmo_internal;

    // the header's edge numbering matches the documented rule
    for (int e = 0; e < 12; ++e) {
        EXPECT_EQ(int(li::kCubeEdgeCorners[e][0]), cube().edge[e][0]) << "edge " << e;
        EXPECT_EQ(int(li::kCubeEdgeCorners[e][1]), cube().edge[e][1]) << "edge " << e;
    }

    for (Conn conn : {Conn::FG26_BG6, Conn::FG6_BG26, Conn::FG18_BG6, Conn::FG6_BG18}) {
        const auto& table = li::caseTable(conn);
        EXPECT_EQ(&table, &li::caseTable(conn)); // generated once
        int failures = 0, tubes = 0, maxTris = 0;
        for (int c = 0; c < 256; ++c) {
            maxTris = std::max(maxTris, int(table[c].count));
            if (c == 0 || c == 255) { EXPECT_EQ(int(table[c].count), 0); continue; }
            const std::string e = checkCase(conn, c, table[c], tubes);
            if (!e.empty() && ++failures <= 8) ADD_FAILURE() << connectivityName(conn) << " case " << c << ": " << e;
        }
        EXPECT_EQ(failures, 0) << connectivityName(conn);
        // tubes: exactly the four body-diagonal pairs for a 26 side
        const bool has26 = conn == Conn::FG26_BG6 || conn == Conn::FG6_BG26;
        EXPECT_EQ(tubes, has26 ? 4 : 0) << connectivityName(conn);
        if (std::getenv("OPENVDB_MOGAC_VERBOSE")) {
            std::cout << "[CaseTableAll] " << connectivityName(conn) << ": max triangles/case " << maxTris
                      << ", tube components " << tubes << std::endl;
        }
    }
}


////////////////////////////////////////
// multiObjectToMesh

namespace {

using MLS = lsmo_test::MLS;
using tools::LabelMesh;
const Conn kAllConn[4] = {Conn::FG26_BG6, Conn::FG6_BG26, Conn::FG18_BG6, Conn::FG6_BG18};

// Doubled triangle area vectors below this length (squared world units)
// count as zero area.
constexpr double kZeroArea = 1e-10;

Vec3d
triNormal(const LabelMesh& m, const Vec3I& t)
{
    const Vec3d a(m.points[t[0]]), b(m.points[t[1]]), c(m.points[t[2]]);
    return (b - a).cross(c - a);
}

// Closed 2-manifold check of lsmo_test::meshTopology, with the defect as
// the failure message.
::testing::AssertionResult
closedManifold(const LabelMesh& m, double dx)
{
    const auto t = lsmo_test::meshTopology(m, dx);
    if (t.closed && t.manifold) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << (t.closed ? "" : "not closed ("
        + std::to_string(t.boundaryEdgeMidpoints.size()) + " boundary edges) ") << t.defect;
}

// Folded triangles: triangles are grouped by the cell holding
// their centroid and by connectivity within the cell into patches. A disk
// patch (Euler characteristic 1) must have every triangle normal along the
// patch's vector area; a tube (an annulus around the body diagonal of two
// in-corners or two out-corners) must have every normal pointing away from
// (in-tube) or towards (out-tube) the diagonal.
template<typename MultiLevelSetT>
size_t
countFolded(const LabelMesh& m, const MultiLevelSetT& mls, int32_t label,
    size_t* tubes = nullptr)
{
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;
    const math::Transform& xf = mls.phi()->transform();
    tree::ValueAccessor<const LabelTreeT> lab(mls.labels()->tree());
    std::map<Coord, std::vector<size_t>> cells;
    for (size_t i = 0; i < m.triangles.size(); ++i) {
        const Vec3I& t = m.triangles[i];
        const Vec3d c = (Vec3d(m.points[t[0]]) + Vec3d(m.points[t[1]]) + Vec3d(m.points[t[2]])) / 3.0;
        cells[Coord::floor(xf.worldToIndex(c))].push_back(i);
    }
    size_t folded = 0;
    for (const auto& cell : cells) {
        const auto& ids = cell.second;
        std::vector<size_t> rep(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) rep[i] = i;
        std::function<size_t(size_t)> find = [&](size_t x) { while (rep[x] != x) x = rep[x]; return x; };
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = i + 1; j < ids.size(); ++j) {
                int shared = 0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        shared += m.triangles[ids[i]][a] == m.triangles[ids[j]][b];
                if (shared >= 2) rep[find(i)] = find(j);
            }
        std::map<size_t, std::vector<size_t>> patches;
        for (size_t i = 0; i < ids.size(); ++i) patches[find(i)].push_back(ids[i]);
        int inMask = 0;
        for (int c = 0; c < 8; ++c)
            if (lab.getValue(cell.first.offsetBy(c & 1, (c >> 1) & 1, (c >> 2) & 1)) == label)
                inMask |= 1 << c;
        for (const auto& p : patches) {
            std::vector<Vec3I> sub;
            for (size_t i : p.second) sub.push_back(m.triangles[i]);
            const int chi = lsmo_test::eulerCharacteristic(sub);
            Vec3d area(0.0);
            for (size_t i : p.second) area += triNormal(m, m.triangles[i]);
            int axisMask = -1;
            if (chi == 0) {
                if (tubes) ++*tubes;
                auto antipodal = [](int mask) {
                    int n = 0, first = -1, second = -1;
                    for (int c = 0; c < 8; ++c) if (mask & (1 << c)) { ++n; (first < 0 ? first : second) = c; }
                    return n == 2 && (first ^ second) == 7;
                };
                axisMask = antipodal(inMask) ? inMask : (antipodal(~inMask & 255) ? (~inMask & 255) : -2);
            }
            for (size_t i : p.second) {
                const Vec3I& t = m.triangles[i];
                const Vec3d n = triNormal(m, t);
                double d = 0.0;
                if (axisMask == -1) {
                    d = n.dot(area);
                } else if (axisMask >= 0) {
                    int a0 = -1, a1 = -1;
                    for (int c = 0; c < 8; ++c) if (axisMask & (1 << c)) (a0 < 0 ? a0 : a1) = c;
                    const Vec3d p0 = xf.indexToWorld(cell.first.offsetBy(a0 & 1, (a0 >> 1) & 1, (a0 >> 2) & 1));
                    const Vec3d p1 = xf.indexToWorld(cell.first.offsetBy(a1 & 1, (a1 >> 1) & 1, (a1 >> 2) & 1));
                    const Vec3d axis = (p1 - p0).unit();
                    const Vec3d cen = (Vec3d(m.points[t[0]]) + Vec3d(m.points[t[1]]) + Vec3d(m.points[t[2]])) / 3.0 - p0;
                    const Vec3d radial = cen - axis * axis.dot(cen);
                    d = n.dot(radial) * (axisMask == inMask ? 1.0 : -1.0);
                }
                if (!(d > 0.0)) ++folded;
            }
        }
    }
    return folded;
}

// Random label volume: labels 1..nLabels drawn per
// block of `block`^3 voxels over [0, n-1]^3, label 0 outside.
// - randomPhi false: the pair built by MultiLevelSet::fromLabelGrid (the
//   reference random label volumes);
// - randomPhi true (stress variant): phi set directly over [-2, n+1]^3,
//   uniform in [0.02, 0.98]*dx at voxels with a differing face neighbor
//   and in [1, 2.9]*dx elsewhere, so the vertex on every cell edge sits at
//   an independent random position (not the zero set of any smooth field).
MLS::Ptr
makeRandomLabels(uint32_t seed, int n, int nLabels, int block, bool randomPhi)
{
    std::mt19937 rng(seed);
    auto xform = math::Transform::createLinearTransform(1.0);
    const int nb = (n + block - 1) / block;
    std::vector<int32_t> blocks(size_t(nb) * nb * nb);
    for (auto& b : blocks) b = int32_t(1 + rng() % uint32_t(nLabels));
    auto labelAt = [&](const Coord& c) -> int32_t {
        for (int a = 0; a < 3; ++a) if (c[a] < 0 || c[a] >= n) return 0;
        return blocks[(size_t(c[0] / block) * nb + c[1] / block) * nb + c[2] / block];
    };
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    lab->setTransform(xform);
    auto la = lab->getAccessor();
    if (!randomPhi) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) la.setValue(Coord(i, j, k), static_cast<MLS::LabelT>(labelAt(Coord(i, j, k))));
        return MLS::fromLabelGrid(*lab, 3);
    }
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    phi->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    for (int i = -2; i <= n + 1; ++i)
        for (int j = -2; j <= n + 1; ++j)
            for (int k = -2; k <= n + 1; ++k) {
                const Coord c(i, j, k);
                bool iface = false;
                for (int f = 0; f < 6; ++f) iface = iface || labelAt(c + offs[f]) != labelAt(c);
                const double u = double(rng()) / 4294967296.0;
                la.setValue(c, static_cast<MLS::LabelT>(labelAt(c)));
                pa.setValue(c, float(iface ? 0.02 + 0.96 * u : 1.0 + 1.9 * u));
            }
    return MLS::Ptr(new MLS(phi, lab, 3));
}

const uint32_t kRandomSeeds[5] = {11u, 22u, 33u, 44u, 55u};
const int kRandomBlock[5] = {1, 1, 1, 2, 3};

// fromLabelGrid on a label grid filled by a predicate over [lo, hi]^3, as a
// MultiLevelSetT (the default MultiLevelSet unless given).
template<typename MultiLevelSetT = MLS>
typename MultiLevelSetT::Ptr
labelFixture(const CoordBBox& box, const std::function<int32_t(const Coord&)>& f)
{
    Int32Grid::Ptr g = Int32Grid::create(0);
    g->setTransform(math::Transform::createLinearTransform(1.0));
    auto acc = g->getAccessor();
    for (auto it = box.begin(); it; ++it) {
        const int32_t l = f(*it);
        if (l != 0) acc.setValue(*it, l);
    }
    return MultiLevelSetT::fromLabelGrid(*g, 3);
}

std::map<int32_t, LabelMesh>
meshWithStats(const MLS& m, const tools::MultiObjectMeshSettings& s,
    tools::lsmo_internal::MeshStats& stats)
{
    return tools::lsmo_internal::contourMultiObject(m, {}, s,
        static_cast<util::NullInterrupter*>(nullptr), &stats);
}

void
printStats(const std::string& name, const tools::lsmo_internal::MeshStats& s)
{
    std::cout << "[Mesh] " << name << ": patches " << s.patches << ", retriangulated "
              << s.retriangulated << ", folded " << s.folded << ", junction-corrected vertices "
              << s.corrected << std::endl;
}

void
expectSameMeshes(const std::map<int32_t, LabelMesh>& a, const std::map<int32_t, LabelMesh>& b)
{
    ASSERT_EQ(a.size(), b.size());
    for (auto ia = a.begin(), ib = b.begin(); ia != a.end(); ++ia, ++ib) {
        ASSERT_EQ(ia->first, ib->first);
        EXPECT_TRUE(ia->second.points == ib->second.points) << "label " << ia->first;
        EXPECT_TRUE(ia->second.triangles == ib->second.triangles) << "label " << ia->first;
        EXPECT_TRUE(ia->second.normals == ib->second.normals) << "label " << ia->first;
    }
}

// Asserts closedness, manifoldness and no folded triangles for every mesh.
void
expectManifold(const std::string& name, const MLS& m, const std::map<int32_t, LabelMesh>& meshes,
    size_t* totalTris = nullptr)
{
    for (const auto& kv : meshes) {
        EXPECT_TRUE(closedManifold(kv.second, m.voxelSize())) << name << " label " << kv.first;
        EXPECT_FALSE(kv.second.triangles.empty()) << name << " label " << kv.first;
        EXPECT_EQ(countFolded(kv.second, m, kv.first), size_t(0)) << name << " label " << kv.first;
        if (totalTris) *totalTris += kv.second.triangles.size();
    }
}

} // namespace


// Closed 2-manifolds, no folded or zero-area triangles.
TEST_F(TestLevelSetMultiObjectMesh, MeshManifold)
{
    tools::MultiObjectMeshSettings s;
    tools::lsmo_internal::MeshStats stats;
    {
        auto m = lsmo_test::makeTwoSpheres(1.0);
        const auto meshes = meshWithStats(*m, s, stats);
        ASSERT_EQ(meshes.size(), size_t(2));
        size_t tris = 0;
        expectManifold("two spheres", *m, meshes, &tris);
        for (const auto& kv : meshes) {
            const auto t = lsmo_test::meshTopology(kv.second, m->voxelSize());
            EXPECT_EQ(t.components, 1);
            EXPECT_EQ(t.euler, 2);
        }
        printStats("two spheres (" + std::to_string(tris) + " triangles)", stats);
    }
    {
        auto m = lsmo_test::Vortex().endState();
        const auto start = std::chrono::steady_clock::now();
        auto meshes = meshWithStats(*m, s, stats);
        std::cout << "[Mesh] vortex end state: multiObjectToMesh "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " s" << std::endl;
        EXPECT_EQ(meshes.size(), size_t(8));
        size_t tris = 0;
        expectManifold("vortex", *m, meshes, &tris);
        printStats("vortex end state (" + std::to_string(tris) + " triangles)", stats);
        // identical output serial and parallel, also with normals
        tools::MultiObjectMeshSettings serial = s, parallel = s;
        serial.grainSize = 0;
        serial.computeNormals = parallel.computeNormals = true;
        parallel.grainSize = 1;
        const auto a = tools::multiObjectToMesh(*m, {}, serial);
        const auto b = tools::multiObjectToMesh(*m, {}, parallel);
        expectSameMeshes(a, b);
        for (const auto& kv : a) {
            EXPECT_EQ(kv.second.normals.size(), kv.second.points.size());
            EXPECT_TRUE(kv.second.triangles == meshes[kv.first].triangles);
        }
    }
    for (int r = 0; r < 5; ++r) {
        auto m = makeRandomLabels(kRandomSeeds[r], 24, 4, kRandomBlock[r], false);
        for (Conn conn : kAllConn) {
            s.connectivity = conn;
            const auto meshes = meshWithStats(*m, s, stats);
            EXPECT_EQ(meshes.size(), size_t(4));
            size_t tris = 0;
            const std::string name = "random " + std::to_string(r) + " " + connectivityName(conn);
            expectManifold(name, *m, meshes, &tris);
            printStats(name + " (" + std::to_string(tris) + " triangles)", stats);
        }
    }
    // Stress variant with an independent random vertex on every cell edge:
    // still closed manifolds without zero-area triangles; folded triangles
    // are reported, not asserted (a twisted 7-vertex loop may admit no
    // unfolded triangulation of its edge vertices).
    for (int r = 0; r < 5; ++r) {
        auto m = makeRandomLabels(kRandomSeeds[r], 24, 4, kRandomBlock[r], true);
        for (Conn conn : kAllConn) {
            s.connectivity = conn;
            const auto meshes = meshWithStats(*m, s, stats);
            EXPECT_EQ(meshes.size(), size_t(4));
            size_t tris = 0, folded = 0;
            const std::string name = "random-phi " + std::to_string(r) + " " + connectivityName(conn);
            for (const auto& kv : meshes) {
                EXPECT_TRUE(closedManifold(kv.second, 1.0)) << name << " label " << kv.first;
                tris += kv.second.triangles.size();
                folded += countFolded(kv.second, *m, kv.first);
            }
            printStats(name + " (" + std::to_string(tris) + " triangles, " + std::to_string(folded)
                + " folded triangles)", stats);
        }
    }
}


namespace {

// Solid torus, R = 8, r = 3, about z (label 1), and its padded domain.
MLS::Ptr
torusFixture()
{
    return labelFixture(CoordBBox(Coord(-13), Coord(13)), [](const Coord& c) {
        const double rho = std::sqrt(double(c[0] * c[0] + c[1] * c[1])) - 8.0;
        return rho * rho + double(c[2] * c[2]) <= 9.0 ? 1 : 0;
    });
}
const CoordBBox kTorusDomain(Coord(-14), Coord(14));

// Hollow sphere: shell 5.5 <= r <= 9 (label 1) around a cavity holding
// label 0 and a ball r <= 3 of label 2.
MLS::Ptr
hollowSphereFixture()
{
    return labelFixture(CoordBBox(Coord(-10), Coord(10)), [](const Coord& c) {
        const double r = c.asVec3d().length();
        return r <= 3.0 ? 2 : (r >= 5.5 && r <= 9.0 ? 1 : 0);
    });
}
const CoordBBox kHollowDomain(Coord(-11), Coord(11));

} // namespace


// Mesh components == foreground components + cavities and
// chi(M) == 2 chi(R) for every label and connectivity pair
// (lsmo_test::verifyLabelTopology), and the explicit numbers of the
// hand-made fixtures.
TEST_F(TestLevelSetMultiObjectMesh, MeshTopology)
{
    struct Fixture
    {
        std::string name;
        MLS::Ptr mls;
        CoordBBox domain;
    };
    std::vector<Fixture> fixtures;
    fixtures.push_back({"torus", torusFixture(), kTorusDomain});
    // two 4^3 cubes sharing only the z-edge line between (3,3,z) and (4,4,z)
    fixtures.push_back({"edge cubes", labelFixture(CoordBBox(Coord(0), Coord(7)), [](const Coord& c) {
        const bool a = c[0] <= 3 && c[1] <= 3 && c[2] <= 3;
        const bool b = c[0] >= 4 && c[1] >= 4 && c[2] <= 3;
        return a || b ? 1 : 0;
    }), CoordBBox(Coord(-2), Coord(9))});
    // two 4^3 cubes sharing only the corner between (3,3,3) and (4,4,4)
    fixtures.push_back({"corner cubes", labelFixture(CoordBBox(Coord(0), Coord(7)), [](const Coord& c) {
        const bool a = c[0] <= 3 && c[1] <= 3 && c[2] <= 3;
        const bool b = c[0] >= 4 && c[1] >= 4 && c[2] >= 4;
        return a || b ? 1 : 0;
    }), CoordBBox(Coord(-2), Coord(9))});
    fixtures.push_back({"hollow sphere", hollowSphereFixture(), kHollowDomain});
    // 2x2x2 block minus an antipodal pair (FG6_BG18 hexagon correction)
    fixtures.push_back({"antipodal hexagon", labelFixture(CoordBBox(Coord(0), Coord(1)), [](const Coord& c) {
        return c == Coord(0) || c == Coord(1) ? 0 : 1;
    }), CoordBBox(Coord(-2), Coord(3))});
    for (int r = 0; r < 5; ++r) {
        fixtures.push_back({"random " + std::to_string(r),
            makeRandomLabels(kRandomSeeds[r], 24, 4, kRandomBlock[r], false), CoordBBox(Coord(-2), Coord(25))});
    }

    // explicit expectations (none for the random fixtures)
    auto expected = [](const std::string& name, Conn conn, int32_t label,
                        lsmo_test::DigitalTopology& t) {
        const int fg = fgOf(conn);
        if (name == "torus") { t = {1, 0, 0}; return true; }
        if (name == "edge cubes") { t = {fg == 6 ? 2 : 1, 0, fg == 6 ? 2 : 1}; return true; }
        if (name == "corner cubes") { t = {fg == 26 ? 1 : 2, 0, fg == 26 ? 1 : 2}; return true; }
        if (name == "hollow sphere") {
            t = label == 1 ? lsmo_test::DigitalTopology{1, 1, 2} : lsmo_test::DigitalTopology{1, 0, 1};
            return true;
        }
        if (name == "antipodal hexagon") {
            t = {1, 0, conn == Conn::FG6_BG26 ? 0 : 1};  // FG6_BG18: filled hexagon
            return true;
        }
        return false;
    };

    for (const Fixture& fx : fixtures) {
        const std::vector<int32_t> present = fx.mls->labelSet();
        for (Conn conn : kAllConn) {
            tools::MultiObjectMeshSettings s;
            s.connectivity = conn;
            const auto meshes = tools::multiObjectToMesh(*fx.mls, {}, s);
            EXPECT_EQ(meshes.size(), present.size()) << fx.name << " " << connectivityName(conn);
            std::ostringstream row;
            row << "[MeshTopology] " << fx.name << " " << connectivityName(conn) << ":";
            for (int32_t l : present) {
                ASSERT_TRUE(meshes.count(l)) << fx.name << " label " << l;
                lsmo_test::DigitalTopology want;
                const bool has = expected(fx.name, conn, l, want);
                const auto rep = lsmo_test::verifyLabelTopology(*fx.mls, l, fx.domain, conn,
                    fx.name + "/" + connectivityName(conn) + "/label" + std::to_string(l),
                    has ? &want : nullptr);
                EXPECT_TOPOLOGY_CONSISTENT(rep);
                // the label's mesh from the all-label call has the same topology
                const auto all = lsmo_test::meshTopology(meshes.at(l));
                EXPECT_EQ(all.components, rep.mesh.components) << fx.name << " label " << l;
                EXPECT_EQ(all.euler, rep.mesh.euler) << fx.name << " label " << l;
                row << " L" << l << " comps " << rep.mesh.components << "=" << rep.region.components
                    << "+" << rep.region.cavities << " chi " << rep.mesh.euler << "=2*"
                    << rep.region.euler;
            }
            if (fx.name.compare(0, 6, "random") != 0 || conn == Conn::FG26_BG6) {
                std::cout << row.str() << std::endl;
            }
        }
    }
}


// lsmo_test::meshTopology and verifyLabelTopology themselves: a closed
// sphere, the same mesh with one triangle removed, the torus and
// hollow-sphere fixtures, a wrong expectation and the debug dump.
TEST_F(TestLevelSetMultiObjectMesh, MeshTopologyUtility)
{
    namespace fs = std::filesystem;
    auto spheres = lsmo_test::makeTwoSpheres(1.0);
    const LabelMesh sphere = tools::multiObjectToMesh(*spheres, {1}).at(1);
    const auto t = lsmo_test::meshTopology(sphere);
    EXPECT_EQ(t.components, 1);
    EXPECT_EQ(t.euler, 2);
    EXPECT_TRUE(t.closed);
    EXPECT_TRUE(t.manifold) << t.defect;
    EXPECT_TRUE(t.boundaryEdgeMidpoints.empty());
    EXPECT_TRUE(t.nonManifoldEdgeMidpoints.empty());
    // one triangle removed: an open disk, its three edges on the boundary
    LabelMesh holed = sphere;
    const Vec3I gone = holed.triangles.back();
    holed.triangles.pop_back();
    const auto h = lsmo_test::meshTopology(holed);
    EXPECT_FALSE(h.closed);
    EXPECT_TRUE(h.manifold) << h.defect;
    EXPECT_EQ(h.components, 1);
    EXPECT_EQ(h.euler, 1);
    ASSERT_EQ(h.boundaryEdgeMidpoints.size(), size_t(3));
    for (int k = 0; k < 3; ++k) {
        const Vec3s mid = (sphere.points[gone[k]] + sphere.points[gone[(k + 1) % 3]]) * 0.5f;
        EXPECT_TRUE(std::find(h.boundaryEdgeMidpoints.begin(), h.boundaryEdgeMidpoints.end(), mid)
            != h.boundaryEdgeMidpoints.end()) << "edge " << k;
    }
    // a triangle added twice: its edges have three triangles
    LabelMesh doubled = sphere;
    doubled.triangles.push_back(doubled.triangles.front());
    const auto d = lsmo_test::meshTopology(doubled);
    EXPECT_FALSE(d.manifold);
    EXPECT_EQ(d.nonManifoldEdgeMidpoints.size(), size_t(3));

    // the fixtures, under every rule
    auto torus = torusFixture();
    auto hollow = hollowSphereFixture();
    const lsmo_test::DigitalTopology solidTorus{1, 0, 0}, shell{1, 1, 2}, ball{1, 0, 1};
    for (Conn conn : kAllConn) {
        const std::string rule = connectivityName(conn);
        const auto rt = lsmo_test::verifyLabelTopology(*torus, 1, kTorusDomain, conn,
            "torus/" + rule, &solidTorus);
        EXPECT_TOPOLOGY_CONSISTENT(rt);
        EXPECT_EQ(rt.mesh.components, 1);
        EXPECT_EQ(rt.mesh.euler, 0);
        const auto rs = lsmo_test::verifyLabelTopology(*hollow, 1, kHollowDomain, conn,
            "hollow/" + rule, &shell);
        EXPECT_TOPOLOGY_CONSISTENT(rs);
        EXPECT_EQ(rs.mesh.components, 2);
        EXPECT_EQ(rs.mesh.euler, 4);
        EXPECT_TOPOLOGY_CONSISTENT(lsmo_test::verifyLabelTopology(*hollow, 2, kHollowDomain, conn,
            "hollow-core/" + rule, &ball));
    }

    // a wrong expectation fails, names the mismatch and writes the dump
    const lsmo_test::ScopedDebugDir debugDir;
    const auto bad = lsmo_test::verifyLabelTopology(*torus, 1, kTorusDomain, Conn::FG26_BG6,
        "wrong-expectation", &ball);
    EXPECT_FALSE(bad.ok);
    EXPECT_NE(bad.message.find("!= expected (components 1, cavities 0, euler 1)"),
        std::string::npos) << bad.message;
    EXPECT_NE(bad.message.find("FG26_BG6"), std::string::npos) << bad.message;
    const fs::path dump = debugDir.path() / "TestLevelSetMultiObjectMesh.MeshTopologyUtility"
        / "wrong-expectation";
    EXPECT_NE(bad.message.find(dump.string()), std::string::npos) << bad.message;
    for (const char* f : {"pair.vdb", "label1.ply", "report.txt"}) {
        EXPECT_TRUE(fs::exists(dump / f)) << (dump / f).string();
    }
    {
        std::ifstream is(dump / "report.txt");
        const std::string report((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        EXPECT_NE(report.find("component 0:"), std::string::npos) << report;
        io::File file((dump / "pair.vdb").string());
        file.open();
        EXPECT_TRUE(file.hasGrid("phi"));
        EXPECT_TRUE(file.hasGrid("labels"));
        file.close();
    }
    // with OPENVDB_MOGAC_DEBUG_DIR set, a passing verification dumps too
    EXPECT_TOPOLOGY_CONSISTENT(lsmo_test::verifyLabelTopology(*torus, 1, kTorusDomain,
        Conn::FG26_BG6, "passing", &solidTorus));
    EXPECT_TRUE(fs::exists(debugDir.path() / "TestLevelSetMultiObjectMesh.MeshTopologyUtility"
        / "passing" / "report.txt"));
}


// Regression: a 2x2x2 cell whose two antipodal out-corners have
// phi = 1e-3*dx (FG6_BG26 tube of the outside region) once made the tube
// retriangulation reuse a chord (a pinched double cone, chi 3). Corner
// values are those of the cell found in the cube-through-spheres run.
TEST_F(TestLevelSetMultiObjectMesh, MeshTubeNoRepeatedChord)
{
    // cell corners at (0..1)^3 + lo; (1,0,0) and (0,1,1) are label 0
    const Coord lo(1, 1, 1);
    const std::map<Coord, float> cell = {
        {Coord(0, 0, 0), 0.0494777f}, {Coord(1, 0, 0), 0.001f},
        {Coord(0, 1, 0), 0.0498429f}, {Coord(1, 1, 0), 0.162809f},
        {Coord(0, 0, 1), 0.257466f},  {Coord(1, 0, 1), 0.472167f},
        {Coord(0, 1, 1), 0.001f},     {Coord(1, 1, 1), 0.239218f}};
    auto xform = math::Transform::createLinearTransform(1.0);
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    // label 1 over [-2, 5]^3 minus the two out-corners; label 0 around it
    const CoordBBox body(Coord(-2), Coord(5));
    for (auto it = CoordBBox(Coord(-4), Coord(7)).begin(); it; ++it) {
        const Coord& p = *it;
        const auto c = cell.find(p - lo);
        const bool out = c != cell.end() && (p - lo == Coord(1, 0, 0) || p - lo == Coord(0, 1, 1));
        const bool in = body.isInside(p) && !out;
        la.setValue(p, in ? 1 : 0);
        float d;
        if (c != cell.end()) {
            d = c->second;
        } else {
            // distance to the body's outer surface, in voxels (staircase)
            int depth = 100;
            for (int a = 0; a < 3; ++a) {
                depth = std::min({depth, std::abs(p[a] - body.min()[a] + (in ? 0 : 1)),
                    std::abs(body.max()[a] - p[a] + (in ? 0 : 1))});
            }
            d = std::min(2.9f, 0.5f + float(in ? depth : std::max(0, depth - 1)));
        }
        pa.setValue(p, d);
    }
    MLS m(phi, lab, 3);
    tools::MultiObjectMeshSettings s;
    s.connectivity = Conn::FG6_BG26;
    tools::lsmo_internal::MeshStats stats;
    const auto meshes = meshWithStats(m, s, stats);
    const auto t = lsmo_test::meshTopology(meshes.at(1));
    std::cout << "[MeshTubeNoRepeatedChord] retriangulated " << stats.retriangulated
              << ", euler " << t.euler << ", components " << t.components << std::endl;
    EXPECT_TRUE(t.closed);
    EXPECT_TRUE(t.manifold) << t.defect;
    EXPECT_TRUE(t.nonManifoldEdgeMidpoints.empty());
    // outer surface (sphere) plus the tube's cavity surface (sphere)
    EXPECT_EQ(t.components, 2);
    EXPECT_EQ(t.euler, 4);
}


// Sphere vertices within 0.1*dx of the analytic surface.
TEST_F(TestLevelSetMultiObjectMesh, MeshGeometrySphere)
{
    for (double dx : {1.0, 0.5}) {
        const double r = 10.0 * dx;
        const Vec3d c(0.31 * dx, -0.17 * dx, 0.23 * dx);
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c), float(dx));
        auto m = MLS::fromSignedGrids(in);
        tools::MultiObjectMeshSettings s;
        s.computeNormals = true;
        const auto meshes = tools::multiObjectToMesh(*m, {}, s);
        ASSERT_EQ(meshes.size(), size_t(1));
        const LabelMesh& mesh = meshes.at(1);
        EXPECT_TRUE(closedManifold(mesh, dx));
        EXPECT_EQ(lsmo_test::meshTopology(mesh, dx).euler, 2);
        double maxErr = 0.0, minDot = 1.0;
        for (size_t i = 0; i < mesh.points.size(); ++i) {
            const Vec3d d = Vec3d(mesh.points[i]) - c;
            maxErr = std::max(maxErr, std::abs(d.length() - r) / dx);
            minDot = std::min(minDot, d.unit().dot(Vec3d(mesh.normals[i])));
        }
        std::cout << "[MeshGeometrySphere] dx " << dx << ": " << mesh.points.size() << " vertices, "
                  << mesh.triangles.size() << " triangles, max vertex error " << maxErr
                  << " dx, min normal . radial " << minDot << std::endl;
        EXPECT_LT(maxErr, 0.1);
        EXPECT_GT(minDot, 0.95);
        // outward winding: positive enclosed volume
        double vol = 0.0;
        for (const Vec3I& t : mesh.triangles) {
            vol += Vec3d(mesh.points[t[0]]).dot(Vec3d(mesh.points[t[1]]).cross(Vec3d(mesh.points[t[2]])));
        }
        vol /= 6.0;
        const double exact = 4.0 / 3.0 * math::pi<double>() * r * r * r;
        EXPECT_NEAR(vol, exact, 0.02 * exact);
    }
}


// Junction correction lowers the max vertex error at triple
// junctions. Asserted on the original triple-junction fixture (junction 0 of the seeded set)
// and on the worst case over the whole set; every junction's
// numbers are printed. Vertices within 3 voxels of the box faces are skipped
// (the box boundary is not part of the analytic partition).
TEST_F(TestLevelSetMultiObjectMesh, MeshJunctionCorrection)
{
    const auto set = lsmo_test::junctionSet();
    double worst[2] = {0.0, 0.0};
    for (size_t n = 0; n < set.size(); ++n) {
        const lsmo_test::TripleJunction& tj = set[n];
        auto m = tj.build();
        double err[2] = {0.0, 0.0};
        for (int corr = 0; corr < 2; ++corr) {
            tools::MultiObjectMeshSettings s;
            s.junctionCorrection = corr == 1;
            tools::lsmo_internal::MeshStats stats;
            const auto meshes = meshWithStats(*m, s, stats);
            ASSERT_EQ(meshes.size(), size_t(3));
            for (const auto& kv : meshes) {
                for (const Vec3s& p : kv.second.points) {
                    const Vec3d w(p);
                    const Vec3d x = m->phi()->transform().worldToIndex(w);
                    const double depth = tj.H - std::max({std::abs(x[0] + 0.5),
                        std::abs(x[1] + 0.5), std::abs(x[2] + 0.5)});
                    if (depth < 3.0) continue; // box faces: not part of the analytic partition
                    err[corr] = std::max(err[corr], std::abs(tj.signedDist(w, kv.first)) / tj.dx);
                }
            }
            if (n == 0) printStats(std::string("junction 0 correction ") + (corr ? "on" : "off"), stats);
        }
        std::cout << "[MeshJunctionCorrection] junction " << n << ": max vertex error off "
                  << err[0] << " dx, on " << err[1] << " dx" << std::endl;
        if (n == 0) { EXPECT_LT(err[1], err[0]) << "test 3 fixture"; }
        worst[0] = std::max(worst[0], err[0]);
        worst[1] = std::max(worst[1], err[1]);
    }
    std::cout << "[MeshJunctionCorrection] worst over " << set.size() << " junctions: off "
              << worst[0] << " dx, on " << worst[1] << " dx" << std::endl;
    EXPECT_LT(worst[1], worst[0]);
}


// Closest-point junction reconstruction (closestPointJunctions, on by
// default, with junctionCorrection on): lowers the worst vertex error over the
// junction set compared with the Godunov estimate, which the option off
// reproduces (recorded before the option existed). Vertices move along their
// edges only: vertex and triangle counts, closedness and Euler characteristic
// are unchanged, and no triangle is folded.
TEST_F(TestLevelSetMultiObjectMesh, MeshClosestPointJunctions)
{
    EXPECT_TRUE(tools::MultiObjectMeshSettings().closestPointJunctions);
    constexpr double kRecordedOffWorst = 0.279997; // worst over the set, option off
    constexpr double kRecordedTolerance = 1e-5;    // printed precision of the record
    const auto set = lsmo_test::junctionSet();
    double worst[2] = {0.0, 0.0};
    for (size_t n = 0; n < set.size(); ++n) {
        const lsmo_test::TripleJunction& tj = set[n];
        auto m = tj.build();
        double err[2] = {0.0, 0.0};
        std::map<int32_t, LabelMesh> meshes[2];
        for (int on = 0; on < 2; ++on) {
            tools::MultiObjectMeshSettings s;
            s.closestPointJunctions = on == 1;
            tools::lsmo_internal::MeshStats stats;
            meshes[on] = meshWithStats(*m, s, stats);
            ASSERT_EQ(meshes[on].size(), size_t(3));
            for (const auto& kv : meshes[on]) {
                EXPECT_TRUE(closedManifold(kv.second, tj.dx)) << "junction " << n;
                for (const Vec3s& p : kv.second.points) {
                    const Vec3d w(p);
                    const Vec3d x = m->phi()->transform().worldToIndex(w);
                    const double depth = tj.H - std::max({std::abs(x[0] + 0.5),
                        std::abs(x[1] + 0.5), std::abs(x[2] + 0.5)});
                    if (depth < 3.0) continue; // box faces: not part of the analytic partition
                    err[on] = std::max(err[on], std::abs(tj.signedDist(w, kv.first)) / tj.dx);
                }
            }
        }
        for (auto a = meshes[0].begin(), b = meshes[1].begin(); a != meshes[0].end(); ++a, ++b) {
            // same vertices (one per crossing edge); folded patches may be
            // split differently, so only the triangle count is compared
            EXPECT_EQ(a->second.triangles.size(), b->second.triangles.size()) << "junction " << n;
            EXPECT_EQ(a->second.points.size(), b->second.points.size()) << "junction " << n;
            EXPECT_EQ(countFolded(b->second, *m, b->first), size_t(0)) << "junction " << n;
            EXPECT_EQ(lsmo_test::meshTopology(a->second, tj.dx).euler,
                lsmo_test::meshTopology(b->second, tj.dx).euler) << "junction " << n;
        }
        std::cout << "[MeshClosestPointJunctions] junction " << n << ": max vertex error off "
                  << err[0] << " dx, on " << err[1] << " dx" << std::endl;
        worst[0] = std::max(worst[0], err[0]);
        worst[1] = std::max(worst[1], err[1]);
    }
    std::cout << "[MeshClosestPointJunctions] worst over " << set.size() << " junctions: off "
              << worst[0] << " dx, on " << worst[1] << " dx" << std::endl;
    EXPECT_NEAR(worst[0], kRecordedOffWorst, kRecordedTolerance);
    EXPECT_LT(worst[1], worst[0]);
}


// Review focus 2 and 3: empty input, label ids, requested labels.
TEST_F(TestLevelSetMultiObjectMesh, MeshEmptyAndLabelIds)
{
    {
        auto xform = math::Transform::createLinearTransform(1.0);
        FloatGrid::Ptr phi = FloatGrid::create(3.0f);
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        MLS empty(phi, lab, 3);
        EXPECT_TRUE(tools::multiObjectToMesh(empty).empty());
        // background-only band: active voxels, all label 0
        for (int i = 0; i < 10; ++i) {
            phi->tree().setValue(Coord(i, 0, 0), 1.0f);
            lab->tree().setValue(Coord(i, 0, 0), 0);
        }
        MLS bgOnly(phi, lab, 3);
        EXPECT_TRUE(tools::multiObjectToMesh(bgOnly).empty());
        EXPECT_TRUE(tools::multiObjectToMesh(bgOnly, {0}).empty());
    }
    // label ids beyond 16 bits: the Int32 variant
    using MLS32 = tools::MultiLevelSet<FloatTree, Int32Tree>;
    const int32_t big = int32_t(1) << 30;
    auto twoBoxes = [big](const Coord& c) -> int32_t {
        if (c[1] > 5 || c[2] > 5) return 0;
        return c[0] <= 5 ? -7 : big; // two touching boxes
    };
    auto m = labelFixture<MLS32>(CoordBBox(Coord(0), Coord(11)), twoBoxes);
    for (Conn conn : kAllConn) {
        tools::MultiObjectMeshSettings s;
        s.connectivity = conn;
        const auto meshes = tools::multiObjectToMesh(*m, {}, s);
        ASSERT_EQ(meshes.size(), size_t(2)) << connectivityName(conn);
        for (int32_t l : {-7, big}) {
            ASSERT_TRUE(meshes.count(l)) << l;
            const LabelMesh& mesh = meshes.at(l);
            const auto t = lsmo_test::meshTopology(mesh, 1.0);
            EXPECT_TRUE(closedManifold(mesh, 1.0)) << l;
            EXPECT_EQ(t.components, 1) << l;
            EXPECT_EQ(t.euler, 2) << l;
            EXPECT_EQ(countFolded(mesh, *m, l), size_t(0)) << l;
        }
    }
    // an interrupter that fires yields an empty result
    struct Stop: public util::NullInterrupter
    {
        bool wasInterrupted(int = -1) override { return true; }
    } stop;
    EXPECT_TRUE(tools::multiObjectToMesh(*m, {}, tools::MultiObjectMeshSettings(), &stop).empty());
    // ... also on the serial path (grainSize <= 0), where no TBB group exists
    tools::MultiObjectMeshSettings serial;
    serial.grainSize = 0;
    EXPECT_TRUE(tools::multiObjectToMesh(*m, {}, serial, &stop).empty());
    // requested labels: exactly those keys (absent ones empty), label 0 ignored
    const auto some = tools::multiObjectToMesh(*m, {big, 5, 0});
    ASSERT_EQ(some.size(), size_t(2));
    EXPECT_FALSE(some.at(big).triangles.empty());
    EXPECT_TRUE(some.at(5).points.empty() && some.at(5).triangles.empty());
    EXPECT_TRUE(some.at(big).triangles == tools::multiObjectToMesh(*m).at(big).triangles);
    // 16-bit labels: a requested label the tree cannot hold has an empty mesh
    // (65537 would truncate to 1)
    {
        constexpr int32_t kAliasOfOne = 65537;
        auto m16 = labelFixture(CoordBBox(Coord(0), Coord(11)), [](const Coord& c) -> int32_t {
            if (c[1] > 5 || c[2] > 5) return 0;
            return c[0] <= 5 ? 1 : 2;
        });
        const auto alias = tools::multiObjectToMesh(*m16, {kAliasOfOne, 1});
        ASSERT_EQ(alias.size(), size_t(2));
        EXPECT_TRUE(alias.at(kAliasOfOne).points.empty()
            && alias.at(kAliasOfOne).triangles.empty());
        EXPECT_FALSE(alias.at(1).triangles.empty());
    }
}


////////////////////////////////////////
// Contouring all labels at once against one extractSignedDistance +
// volumeToMesh per label, on the vortex end state. The timings are
// informational only (wall-clock comparisons are unreliable on shared
// machines); only the output is checked.
TEST_F(TestLevelSetMultiObjectMesh, MeshBenchmark)
{
    const lsmo_test::Vortex vortex;
    auto m = vortex.endState();
    ASSERT_TRUE(m);
    const std::vector<int32_t> labels = m->labelSet();
    ASSERT_GE(labels.size(), size_t(2));

    using Clock = std::chrono::steady_clock;
    size_t multiTris = 0, perLabelPolys = 0;
    double multiTime = 1e30, perLabelTime = 1e30;
    for (int run = 0; run < 3; ++run) {
        auto t0 = Clock::now();
        const auto meshes = tools::multiObjectToMesh(*m);
        multiTime = std::min(multiTime, std::chrono::duration<double>(Clock::now() - t0).count());
        multiTris = 0;
        for (const auto& kv : meshes) multiTris += kv.second.triangles.size();

        t0 = Clock::now();
        size_t polys = 0;
        for (int32_t l : labels) {
            FloatGrid::Ptr g = m->extractSignedDistance(l);
            std::vector<Vec3s> points;
            std::vector<Vec3I> tris;
            std::vector<Vec4I> quads;
            tools::volumeToMesh(*g, points, tris, quads, 0.0);
            polys += tris.size() + quads.size();
        }
        perLabelTime = std::min(perLabelTime,
            std::chrono::duration<double>(Clock::now() - t0).count());
        perLabelPolys = polys;
    }
    std::cout << "MeshBenchmark: " << labels.size() << " labels; multiObjectToMesh "
              << multiTime << " s (" << multiTris << " triangles); per-label extract + "
              << "volumeToMesh " << perLabelTime << " s (" << perLabelPolys
              << " triangles+quads); ratio " << perLabelTime / multiTime << std::endl;
    EXPECT_GT(multiTris, size_t(0));
}
