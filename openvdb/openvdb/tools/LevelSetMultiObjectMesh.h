// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObjectMesh.h
///
/// @brief Topology-constrained contouring of multi-object level sets
/// (MultiLevelSet, see LevelSetMultiObject.h): every label is meshed into a
/// closed, consistently wound 2-manifold whose topology matches the label's
/// digital voxel region under a chosen (foreground, background) adjacency
/// pair, MultiObjectConnectivity.
///
/// This header currently provides the settings struct and the
/// connectivity-consistent marching-cubes case tables (spec 6.3 step 3).
///
/// @par Cube numbering (independent of VolumeToMesh.h)
/// A cell is the 2x2x2 block of voxel centres with minimum corner ijk.
/// - Corner bit i (0..7) is the voxel ijk + (i&1, (i>>1)&1, (i>>2)&1); a case
///   index has bit i set when corner i lies in the label ("in-corner").
/// - Edges 0-3 run along x, 4-7 along y, 8-11 along z. Edge 4a+k (axis a) has
///   its two free coordinates, in ascending axis order, equal to
///   (k&1, k>>1); kCubeEdgeCorners lists (lower corner, upper corner):
///   @code
///   x: e0 (0,1)  e1 (2,3)  e2 (4,5)  e3 (6,7)
///   y: e4 (0,2)  e5 (1,3)  e6 (4,6)  e7 (5,7)
///   z: e8 (0,4)  e9 (1,5)  e10 (2,6) e11 (3,7)
///   @endcode
/// - Face f = 2a + s holds the corners whose coordinate a equals s; its
///   outward normal is (2s-1) along axis a.
///
/// @par Case-table rule
/// Triangles are triples of edge ids, counter-clockwise seen from outside
/// the label (normal from in-corners towards out-corners).
/// - Face rule: on a face with exactly two diagonal in-corners the in-pair is
///   joined (boundary segments cut off the two out-corners) when the
///   foreground adjacency is 18 or 26, and separated (segments cut off the
///   in-corners) when it is 6. Faces with two crossing edges have the single
///   segment between them. Since the segments depend only on the face's four
///   corner bits, neighbouring cells always agree.
/// - Interior rule: two corners on a body diagonal that are not otherwise
///   connected are joined by a tube when their side uses 26-adjacency
///   (in-pairs for FG26_BG6, out-pairs for FG6_BG26), separated otherwise.
/// Within a cell the in-corners (out-corners) joined by the local surface
/// are exactly the components of the in-corners (out-corners) under the
/// cube-restricted foreground (background) adjacency, where 6 = cube edges,
/// 18 = + face diagonals and 26 = + body diagonals.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/LevelSetMultiObject.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Settings of the multi-label mesher (spec 6.2).
struct MultiObjectMeshSettings
{
    /// Digital (foreground, background) adjacency the meshes are consistent with.
    MultiObjectConnectivity connectivity = MultiObjectConnectivity::FG26_BG6;
    /// Snap vertices at junctions of three or more labels (spec 6.4).
    bool junctionCorrection = true;
    /// Also output per-vertex normals.
    bool computeNormals = false;
};


namespace lsmo_internal {

/// Edge e of the cell joins corners kCubeEdgeCorners[e][0] < [e][1] (see the
/// file documentation for the numbering).
constexpr uint8_t kCubeEdgeCorners[12][2] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7},   // along x
    {0, 2}, {1, 3}, {4, 6}, {5, 7},   // along y
    {0, 4}, {1, 5}, {2, 6}, {3, 7}};  // along z

/// Triangulation of one case: @c count triangles of edge ids (0..11),
/// counter-clockwise seen from outside the label.
struct CaseTriangulation
{
    uint8_t count = 0;
    std::array<std::array<uint8_t, 3>, 16> tris{};
};

using CaseTable = std::array<CaseTriangulation, 256>;


/// @brief Builds the 256-case table for adjacency pair (fg, bg), each in
/// {6, 18, 26}, from the face and interior rules of the file documentation.
///
/// Per case: (1) directed boundary segments on each face (face rule),
/// oriented so that faceNormal x direction points from in to out;
/// (2) segments chained into closed loops (each crossing edge lies on two
/// faces); (3) the corners are grouped into regions connected along the cube
/// surface, each loop separating one in-region from one out-region; regions
/// of one side that must be joined under that side's adjacency (only body
/// diagonals under 26) are connected by replacing their two loop disks,
/// which bound a common region of the other side, with a triangle-strip
/// tube; (4) every other loop becomes a disk, a fan from a vertex chosen so
/// that no interior edge lies in a cube face. Throws ValueError if a case
/// cannot be resolved or needs more than 16 triangles.
inline CaseTable
generateCaseTable(int fg, int bg)
{
    // integer geometry: coordinates doubled so edge midpoints are integral
    struct V { int x[3]; };
    auto cornerPos = [](int c) { return V{{2 * (c & 1), 2 * ((c >> 1) & 1), 2 * ((c >> 2) & 1)}}; };
    auto midPos = [&](int e) {
        const V a = cornerPos(kCubeEdgeCorners[e][0]), b = cornerPos(kCubeEdgeCorners[e][1]);
        return V{{(a.x[0] + b.x[0]) / 2, (a.x[1] + b.x[1]) / 2, (a.x[2] + b.x[2]) / 2}};
    };
    auto edgeBetween = [](int a, int b) {
        for (int e = 0; e < 12; ++e)
            if ((kCubeEdgeCorners[e][0] == a && kCubeEdgeCorners[e][1] == b) ||
                (kCubeEdgeCorners[e][0] == b && kCubeEdgeCorners[e][1] == a)) return e;
        return -1;
    };
    // faces: f = 2a + s, corners in cyclic order
    int faceCorners[6][4];
    int edgeFaces[12] = {0};
    for (int a = 0; a < 3; ++a) {
        const int u = a == 0 ? 1 : 0, v = a == 2 ? 1 : 2;
        for (int s = 0; s < 2; ++s) {
            const int f = 2 * a + s, base = s << a;
            faceCorners[f][0] = base;
            faceCorners[f][1] = base | (1 << u);
            faceCorners[f][2] = base | (1 << u) | (1 << v);
            faceCorners[f][3] = base | (1 << v);
            for (int j = 0; j < 4; ++j)
                edgeFaces[edgeBetween(faceCorners[f][j], faceCorners[f][(j + 1) % 4])] |= 1 << f;
        }
    }
    auto adjacent = [](int a, int b, int n) {
        const int d = ((a ^ b) & 1) + (((a ^ b) >> 1) & 1) + (((a ^ b) >> 2) & 1);
        return d == 1 || (d == 2 && n >= 18) || (d == 3 && n == 26);
    };
    auto fail = [&](int caseIdx, const char* what) {
        std::ostringstream os;
        os << "multi-object case table (" << fg << "," << bg << "), case " << caseIdx << ": " << what;
        OPENVDB_THROW(ValueError, os.str());
    };

    CaseTable table;
    for (int caseIdx = 1; caseIdx < 255; ++caseIdx) {
        auto isIn = [&](int c) { return ((caseIdx >> c) & 1) != 0; };
        auto inCorner = [&](int e) {
            return isIn(kCubeEdgeCorners[e][0]) ? kCubeEdgeCorners[e][0] : kCubeEdgeCorners[e][1];
        };
        auto outCorner = [&](int e) {
            return isIn(kCubeEdgeCorners[e][0]) ? kCubeEdgeCorners[e][1] : kCubeEdgeCorners[e][0];
        };

        // corner regions along the cube surface (union-find)
        int rep[8];
        for (int i = 0; i < 8; ++i) rep[i] = i;
        auto find = [&](int x) { while (rep[x] != x) x = rep[x]; return x; };
        auto unite = [&](int a, int b) { rep[find(a)] = find(b); };
        for (int e = 0; e < 12; ++e)
            if (isIn(kCubeEdgeCorners[e][0]) == isIn(kCubeEdgeCorners[e][1]))
                unite(kCubeEdgeCorners[e][0], kCubeEdgeCorners[e][1]);

        // (1) directed face segments
        int next[12];
        for (int e = 0; e < 12; ++e) next[e] = -1;
        auto addSegment = [&](int f, int e0, int e1) {
            // orient so that faceNormal x (m1 - m0) points from in to out
            const V m0 = midPos(e0), m1 = midPos(e1);
            const int a = f / 2, sgn = (f & 1) ? 1 : -1;
            const int t[3] = {m1.x[0] - m0.x[0], m1.x[1] - m0.x[1], m1.x[2] - m0.x[2]};
            int side[3] = {0, 0, 0};             // (sgn * axis_a) x t
            const int b = (a + 1) % 3, c = (a + 2) % 3;
            side[c] = sgn * t[b];
            side[b] = -sgn * t[c];
            const V pi = cornerPos(inCorner(e0)), po = cornerPos(outCorner(e0));
            int d = 0;
            for (int k = 0; k < 3; ++k) d += side[k] * (po.x[k] - pi.x[k]);
            if (d == 0) fail(caseIdx, "degenerate face segment");
            if (d < 0) std::swap(e0, e1);
            if (next[e0] != -1) fail(caseIdx, "edge with two outgoing segments");
            next[e0] = e1;
        };
        for (int f = 0; f < 6; ++f) {
            const int* fc = faceCorners[f];
            int crossing[4], nc = 0;
            for (int j = 0; j < 4; ++j)
                if (isIn(fc[j]) != isIn(fc[(j + 1) % 4]))
                    crossing[nc++] = edgeBetween(fc[j], fc[(j + 1) % 4]);
            if (nc == 2) {
                addSegment(f, crossing[0], crossing[1]);
            } else if (nc == 4) {
                // ambiguous face: fg 18/26 joins the in-pair (cut off the
                // out-corners), fg 6 joins the out-pair (cut off the in-corners)
                const bool cutIn = (fg == 6);
                for (int j = 0; j < 4; ++j) {
                    if (isIn(fc[j]) != cutIn) continue;
                    addSegment(f, edgeBetween(fc[(j + 3) % 4], fc[j]),
                        edgeBetween(fc[j], fc[(j + 1) % 4]));
                }
                // the corners not cut off are joined across the face
                const int keep = cutIn ? 0 : 1;
                for (int j = 0; j < 4; ++j)
                    if (int(isIn(fc[j])) == keep) unite(fc[j], fc[(j + 2) % 4]);
            }
        }

        // (2) loops
        struct Loop { std::vector<int> v; int inRegion, outRegion; bool disk = true; };
        std::vector<Loop> loops;
        {
            bool seen[12] = {false};
            for (int e = 0; e < 12; ++e) {
                if (next[e] < 0 || seen[e]) continue;
                Loop loop;
                for (int cur = e; !seen[cur]; cur = next[cur]) {
                    if (next[cur] < 0) fail(caseIdx, "open loop");
                    seen[cur] = true;
                    loop.v.push_back(cur);
                }
                if (loop.v.front() != next[loop.v.back()]) fail(caseIdx, "malformed loop");
                loop.inRegion = find(inCorner(e));
                loop.outRegion = find(outCorner(e));
                loops.push_back(loop);
            }
        }
        // crossing edges must all be covered
        for (int e = 0; e < 12; ++e)
            if ((isIn(kCubeEdgeCorners[e][0]) != isIn(kCubeEdgeCorners[e][1])) != (next[e] >= 0))
                fail(caseIdx, "crossing edge without segment");

        std::vector<std::array<int, 3>> tris;

        // (3) tubes joining same-side regions required by that side's adjacency
        auto dist2 = [&](int e0, int e1) {
            const V a = midPos(e0), b = midPos(e1);
            int d = 0;
            for (int k = 0; k < 3; ++k) d += (a.x[k] - b.x[k]) * (a.x[k] - b.x[k]);
            return d;
        };
        auto addTube = [&](const std::vector<int>& A, const std::vector<int>& B) {
            // Strip between loop A (walked forward) and loop B (walked
            // backward); both keep their disk orientation, which is the
            // boundary orientation the annulus induces.
            const int m = int(A.size()), n = int(B.size());
            int k = 0;
            for (int j = 1; j < n; ++j) if (dist2(A[0], B[j]) < dist2(A[0], B[k])) k = j;
            int i = 0, j = 0; // steps taken along A and B
            while (i < m || j < n) {
                const int a0 = A[i % m], a1 = A[(i + 1) % m];
                const int b0 = B[((k - j) % n + n) % n], b1 = B[((k - j - 1) % n + n) % n];
                const bool advanceA = j >= n || (i < m && dist2(a1, b0) <= dist2(a0, b1));
                if (advanceA) { tris.push_back({a0, a1, b0}); ++i; }
                else          { tris.push_back({b1, b0, a0}); ++j; }
            }
        };
        for (int side = 0; side < 2; ++side) {
            const bool in = side == 0;
            const int n = in ? fg : bg;
            for (int a = 0; a < 8; ++a)
                for (int b = a + 1; b < 8; ++b) {
                    if (isIn(a) != in || isIn(b) != in || !adjacent(a, b, n)) continue;
                    if (find(a) == find(b)) continue;
                    // two still-disk loops bounding a's and b's regions and a
                    // common region of the other side
                    int la = -1, lb = -1;
                    for (size_t p = 0; p < loops.size() && lb < 0; ++p)
                        for (size_t q = 0; q < loops.size() && lb < 0; ++q) {
                            if (p == q || !loops[p].disk || !loops[q].disk) continue;
                            const Loop &P = loops[p], &Q = loops[q];
                            const int pr = in ? P.inRegion : P.outRegion, qr = in ? Q.inRegion : Q.outRegion;
                            const int po = in ? P.outRegion : P.inRegion, qo = in ? Q.outRegion : Q.inRegion;
                            if (find(pr) == find(a) && find(qr) == find(b) && find(po) == find(qo)) {
                                la = int(p); lb = int(q);
                            }
                        }
                    if (lb < 0) fail(caseIdx, "no loop pair to join by a tube");
                    loops[la].disk = loops[lb].disk = false;
                    addTube(loops[la].v, loops[lb].v);
                    unite(a, b);
                }
        }

        // (4) disks: fan from a vertex none of whose chords lies in a face;
        // among those, the apex whose fan (with midpoint vertices) best keeps
        // every triangle normal along the loop's in-to-out direction
        for (const Loop& loop : loops) {
            if (!loop.disk) continue;
            const int n = int(loop.v.size());
            // in-to-out direction, scaled to integers: |I| sum(out) - |O| sum(in)
            int sumIn[3] = {0, 0, 0}, sumOut[3] = {0, 0, 0}, nIn = 0, nOut = 0;
            {
                int seenC = 0;
                for (int e : loop.v) {
                    for (int c : {inCorner(e), outCorner(e)}) {
                        if (seenC & (1 << c)) continue;
                        seenC |= 1 << c;
                        const V p = cornerPos(c);
                        int* s = isIn(c) ? sumIn : sumOut;
                        (isIn(c) ? nIn : nOut) += 1;
                        for (int k = 0; k < 3; ++k) s[k] += p.x[k];
                    }
                }
            }
            int dir[3];
            for (int k = 0; k < 3; ++k) dir[k] = nIn * sumOut[k] - nOut * sumIn[k];
            auto alignment = [&](int e0, int e1, int e2) {
                const V a = midPos(e0), b = midPos(e1), c = midPos(e2);
                const int u[3] = {b.x[0] - a.x[0], b.x[1] - a.x[1], b.x[2] - a.x[2]};
                const int w[3] = {c.x[0] - a.x[0], c.x[1] - a.x[1], c.x[2] - a.x[2]};
                const int nrm[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0]};
                return nrm[0] * dir[0] + nrm[1] * dir[1] + nrm[2] * dir[2];
            };
            int apex = -1, bestScore = 0;
            for (int r = 0; r < n; ++r) {
                bool ok = true;
                for (int s = 2; s < n - 1; ++s)
                    if (edgeFaces[loop.v[r]] & edgeFaces[loop.v[(r + s) % n]]) ok = false;
                if (!ok) continue;
                int score = std::numeric_limits<int>::max();
                for (int s = 1; s + 1 < n; ++s)
                    score = std::min(score, alignment(loop.v[r], loop.v[(r + s) % n],
                        loop.v[(r + s + 1) % n]));
                if (apex < 0 || score > bestScore) { apex = r; bestScore = score; }
            }
            if (apex < 0) fail(caseIdx, "no valid fan apex");
            for (int s = 1; s + 1 < n; ++s)
                tris.push_back({loop.v[apex], loop.v[(apex + s) % n], loop.v[(apex + s + 1) % n]});
        }

        if (tris.size() > 16) fail(caseIdx, "more than 16 triangles");
        CaseTriangulation& ct = table[caseIdx];
        ct.count = uint8_t(tris.size());
        for (size_t t = 0; t < tris.size(); ++t)
            for (int k = 0; k < 3; ++k) ct.tris[t][k] = uint8_t(tris[t][k]);
    }
    return table;
}


/// @brief The case table for a connectivity pair, generated on first use
/// (thread-safe function-local statics).
inline const CaseTable&
caseTable(MultiObjectConnectivity conn)
{
    switch (conn) {
        case MultiObjectConnectivity::FG26_BG6: { static const CaseTable t = generateCaseTable(26, 6); return t; }
        case MultiObjectConnectivity::FG6_BG26: { static const CaseTable t = generateCaseTable(6, 26); return t; }
        case MultiObjectConnectivity::FG18_BG6: { static const CaseTable t = generateCaseTable(18, 6); return t; }
        case MultiObjectConnectivity::FG6_BG18: { static const CaseTable t = generateCaseTable(6, 18); return t; }
    }
    OPENVDB_THROW(ValueError, "unknown MultiObjectConnectivity");
}

} // namespace lsmo_internal

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED
