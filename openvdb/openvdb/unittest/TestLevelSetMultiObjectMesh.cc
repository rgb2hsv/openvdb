// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObjectMesh.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
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
// LevelSetMultiObjectMesh.h (ruling R5) rather than taken from the generator:
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

std::string
connName(Conn c)
{
    switch (c) {
        case Conn::FG26_BG6: return "FG26_BG6";
        case Conn::FG6_BG26: return "FG6_BG26";
        case Conn::FG18_BG6: return "FG18_BG6";
        case Conn::FG6_BG18: return "FG6_BG18";
    }
    return "?";
}

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
        const int f = [&] { for (int g = 0; g < 6; ++g)
            if (C.faceMask[s.first] & C.faceMask[s.second] & (1 << g)) return g; return -1; }();
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
            if (!e.empty() && ++failures <= 8) ADD_FAILURE() << connName(conn) << " case " << c << ": " << e;
        }
        EXPECT_EQ(failures, 0) << connName(conn);
        // tubes: exactly the four body-diagonal pairs for a 26 side
        const bool has26 = conn == Conn::FG26_BG6 || conn == Conn::FG6_BG26;
        EXPECT_EQ(tubes, has26 ? 4 : 0) << connName(conn);
        std::cout << "[CaseTableAll] " << connName(conn) << ": max triangles/case " << maxTris
                  << ", tube components " << tubes << std::endl;
    }
}
