#ifndef DTK_LINEAR_BVH_HPP
#define DTK_LINEAR_BVH_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Pair.hpp>

#include <array>
#include <list>
#include <vector>

// From Tero Karras
// (https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/)

namespace DataTransferKit
{
// Axis-Aligned Bounding Box
struct AABB
{
    using size_type = std::array<double, 6>::size_type;
    AABB() = default;
    AABB( std::array<double, 6> minmax )
        : _minmax( minmax )
    {
    }
    AABB &operator=( std::array<double, 6> const &minmax )
    {
        _minmax = minmax;
        return *this;
    }
    double &operator[]( size_type i ) { return _minmax[i]; }
    double const &operator[]( size_type i ) const { return _minmax[i]; }
    std::array<double, 6> _minmax = {{
        Kokkos::Details::ArithTraits<double>::max(),
        -Kokkos::Details::ArithTraits<double>::max(),
        Kokkos::Details::ArithTraits<double>::max(),
        -Kokkos::Details::ArithTraits<double>::max(),
        Kokkos::Details::ArithTraits<double>::max(),
        -Kokkos::Details::ArithTraits<double>::max(),
    }};
    friend std::ostream &operator<<( std::ostream &os, AABB const &aabb )
    {
        os << "{";
        for ( int d = 0; d < 3; ++d )
            os << " [" << aabb._minmax[2 * d + 0] << ", "
               << aabb._minmax[2 * d + 1] << "],";
        os << "}";
        return os;
    }
};
struct Node
{
    virtual ~Node() = default;
    Node *parent = nullptr;
    Kokkos::pair<Node *, Node *> children = {nullptr, nullptr};
    AABB bounding_box;
};
// Bounding Volume Hierarchy
struct BVH
{
    BVH( AABB const *bounding_boxes, int n );
    bool isLeaf( Node const *node ) const;
    int getObjectIdx( Node const *leaf_node ) const;
    Node *getRoot();
    Node const *getRoot() const;
    std::vector<Node> _leaf_nodes;
    std::vector<Node> _internal_nodes;
    std::vector<int> _sorted_indices;
    AABB _scene_bounding_box; // don't actually really need to store it
};

} // end namespace DataTransferKit

#endif
