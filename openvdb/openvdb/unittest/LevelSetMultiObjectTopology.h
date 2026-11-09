// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/LevelSetMultiObjectTopology.h
///
/// @brief Test-only digital topology of one label of a label grid: number of
/// foreground components, enclosed background cavities and the Euler
/// characteristic of the digital region (spec section 4.7, section 8 tests
/// 12 and 15). Header-only; shared by TestLevelSetMultiObject.cc and
/// TestLevelSetMultiObjectMesh.cc.

#ifndef OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>

#include <cstdint>
#include <vector>

namespace lsmo_test {

struct DigitalTopology
{
    int components = 0; ///< foreground components (fg adjacency)
    int cavities = 0;   ///< background components (bg adjacency) not touching the domain boundary
    int euler = 0;      ///< Euler characteristic of the region (ball 1, hollow shell 2, solid torus 0)
};

inline bool operator==(const DigitalTopology& a, const DigitalTopology& b)
{
    return a.components == b.components && a.cavities == b.cavities && a.euler == b.euler;
}

namespace detail {

/// Squared-length bound of an adjacency: 6 -> 1, 18 -> 2, 26 -> 3.
inline int adjBound(int n) { return n == 6 ? 1 : (n == 18 ? 2 : 3); }

inline void connectivityPair(openvdb::tools::MultiObjectConnectivity c, int& fg, int& bg)
{
    using C = openvdb::tools::MultiObjectConnectivity;
    switch (c) {
        case C::FG26_BG6: fg = 26; bg = 6; break;
        case C::FG6_BG26: fg = 6; bg = 26; break;
        case C::FG18_BG6: fg = 18; bg = 6; break;
        case C::FG6_BG18: fg = 6; bg = 18; break;
    }
}

/// Dense boolean volume over a bounding box; outside reads false.
struct Volume
{
    openvdb::Coord lo;
    int nx, ny, nz;
    std::vector<char> v;
    explicit Volume(const openvdb::CoordBBox& b)
        : lo(b.min()), nx(b.dim().x()), ny(b.dim().y()), nz(b.dim().z())
        , v(size_t(nx) * ny * nz, 0) {}
    size_t idx(int i, int j, int k) const { return (size_t(k) * ny + j) * nx + i; }
    bool in(int i, int j, int k) const
    {
        return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz;
    }
    bool at(int i, int j, int k) const { return in(i, j, k) && v[idx(i, j, k)]; }
};

/// Components of the voxels with value @a want under adjacency @a n;
/// @a touching counts those touching the box boundary.
inline int components(const Volume& vol, bool want, int n, int& touching)
{
    const int bound = adjBound(n);
    std::vector<char> seen(vol.v.size(), 0);
    std::vector<size_t> stack;
    int count = 0;
    touching = 0;
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        const size_t s = vol.idx(i, j, k);
        if (bool(vol.v[s]) != want || seen[s]) continue;
        ++count;
        bool edge = false;
        seen[s] = 1;
        stack.assign(1, s);
        while (!stack.empty()) {
            const size_t c = stack.back();
            stack.pop_back();
            const int ci = int(c % vol.nx), cj = int((c / vol.nx) % vol.ny),
                      ck = int(c / (size_t(vol.nx) * vol.ny));
            if (ci == 0 || cj == 0 || ck == 0 || ci == vol.nx - 1 || cj == vol.ny - 1
                || ck == vol.nz - 1) edge = true;
            for (int c2 = -1; c2 <= 1; ++c2)
            for (int b = -1; b <= 1; ++b)
            for (int a = -1; a <= 1; ++a) {
                const int d2 = a * a + b * b + c2 * c2;
                if (d2 == 0 || d2 > bound) continue;
                const int qi = ci + a, qj = cj + b, qk = ck + c2;
                if (!vol.in(qi, qj, qk)) continue;
                const size_t q = vol.idx(qi, qj, qk);
                if (bool(vol.v[q]) != want || seen[q]) continue;
                seen[q] = 1;
                stack.push_back(q);
            }
        }
        if (edge) ++touching;
    }
    return count;
}

/// Euler characteristic of the union of closed unit cubes (26-adjacency).
inline long eulerClosedCubes(const Volume& vol)
{
    long V = 0, E = 0, F = 0, C = 0;
    // lattice point (i,j,k) is the low corner of voxel (i,j,k)
    for (int k = 0; k <= vol.nz; ++k)
    for (int j = 0; j <= vol.ny; ++j)
    for (int i = 0; i <= vol.nx; ++i) {
        bool vert = false;
        for (int c = 0; c < 2; ++c) for (int b = 0; b < 2; ++b) for (int a = 0; a < 2; ++a)
            vert |= vol.at(i - a, j - b, k - c);
        V += vert;
        // edges from (i,j,k) along x, y, z
        E += vol.at(i, j, k) || vol.at(i, j - 1, k) || vol.at(i, j, k - 1) || vol.at(i, j - 1, k - 1);
        E += vol.at(i, j, k) || vol.at(i - 1, j, k) || vol.at(i, j, k - 1) || vol.at(i - 1, j, k - 1);
        E += vol.at(i, j, k) || vol.at(i - 1, j, k) || vol.at(i, j - 1, k) || vol.at(i - 1, j - 1, k);
        // faces with low corner (i,j,k), normal x, y, z
        F += vol.at(i, j, k) || vol.at(i - 1, j, k);
        F += vol.at(i, j, k) || vol.at(i, j - 1, k);
        F += vol.at(i, j, k) || vol.at(i, j, k - 1);
        C += vol.at(i, j, k);
    }
    return V - E + F - C;
}

/// Euler characteristic of the face-connected complex (6-adjacency):
/// voxels, face-adjacent pairs, full 2x2 squares, full 2x2x2 blocks.
inline long eulerFaceComplex(const Volume& vol)
{
    long V = 0, E = 0, F = 0, C = 0;
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        if (!vol.at(i, j, k)) continue;
        ++V;
        const bool x = vol.at(i + 1, j, k), y = vol.at(i, j + 1, k), z = vol.at(i, j, k + 1);
        E += x + y + z;
        const bool xy = x && y && vol.at(i + 1, j + 1, k);
        const bool xz = x && z && vol.at(i + 1, j, k + 1);
        const bool yz = y && z && vol.at(i, j + 1, k + 1);
        F += xy + xz + yz;
        C += xy && z && vol.at(i, j + 1, k + 1) && vol.at(i + 1, j, k + 1)
             && vol.at(i + 1, j + 1, k + 1);
    }
    return V - E + F - C;
}

/// Number of 2x2x2 blocks in which exactly two voxels of value @a want lie
/// at opposite corners. With want = true (18-adjacency) the lattice point
/// they share is split in two; with want = false (6-adjacency paired with
/// 18) the six foreground voxels around the unconnected background pair
/// enclose no tunnel, so their hexagon is filled.
inline long antipodalBlocks(const Volume& vol, bool want)
{
    long n = 0;
    for (int k = -1; k < vol.nz; ++k)
    for (int j = -1; j < vol.ny; ++j)
    for (int i = -1; i < vol.nx; ++i) {
        int count = 0, first = -1, second = -1;
        for (int c = 0; c < 8; ++c) {
            if (vol.at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)) == want) {
                ++count;
                (first < 0 ? first : second) = c;
            }
        }
        if (count == 2 && (first ^ second) == 7) ++n;
    }
    return n;
}

} // namespace detail

/// @brief Digital topology of the region {labels == @a label} inside
/// @a domain (voxels outside @a domain are ignored).
/// @details components: foreground components under the foreground
/// adjacency; cavities: background components under the background adjacency
/// that do not touch the domain boundary (pad the domain by at least one
/// voxel); euler: Euler characteristic of the region's cell complex: union
/// of closed cubes for 26-adjacency; the same with lattice points shared
/// only by two opposite voxels of a 2x2x2 block split for 18-adjacency; the
/// face-connected complex (voxels, face pairs, full 2x2 squares, full 2x2x2
/// blocks) for 6-adjacency paired with 26; and that plus one filled hexagon
/// per 2x2x2 block missing only two opposite voxels for 6 paired with 18.
inline DigitalTopology
digitalTopology(const openvdb::Int32Grid& labels, int32_t label, const openvdb::CoordBBox& domain,
    openvdb::tools::MultiObjectConnectivity conn)
{
    int fg = 26, bg = 6;
    detail::connectivityPair(conn, fg, bg);
    detail::Volume vol(domain);
    openvdb::tree::ValueAccessor<const openvdb::Int32Tree> acc(labels.tree());
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        vol.v[vol.idx(i, j, k)] = acc.getValue(vol.lo.offsetBy(i, j, k)) == label;
    }
    DigitalTopology t;
    int touching = 0;
    t.components = detail::components(vol, true, fg, touching);
    const int bgCount = detail::components(vol, false, bg, touching);
    t.cavities = bgCount - touching;
    if (fg == 6 && bg == 26) t.euler = int(detail::eulerFaceComplex(vol));
    else if (fg == 6) t.euler = int(detail::eulerFaceComplex(vol) + detail::antipodalBlocks(vol, false));
    else if (fg == 18) t.euler = int(detail::eulerClosedCubes(vol) + detail::antipodalBlocks(vol, true));
    else t.euler = int(detail::eulerClosedCubes(vol));
    return t;
}

} // namespace lsmo_test

#endif // OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED
