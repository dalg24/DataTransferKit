#ifndef DTK_LINEAR_BVH_HPP
#define DTK_LINEAR_BVH_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>

#include <vector> // TODO: replace with Kokkos::View

namespace DataTransferKit
{
// Axis-Aligned Bounding Box
struct AABB
{
    using Array = Kokkos::Array<double, 6>;
    using SizeType = Array::size_type;
    AABB() = default;
    AABB( Array const &minmax )
        : _minmax( minmax )
    {
    }
    AABB &operator=( Array const &minmax )
    {
        _minmax = minmax;
        return *this;
    }
    double &operator[]( SizeType i ) { return _minmax[i]; }
    double const &operator[]( SizeType i ) const { return _minmax[i]; }
    Array _minmax = {{
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
    }};
    friend std::ostream &operator<<( std::ostream &os, AABB const &aabb )
    {
        os << "{";
        for ( int d = 0; d < 3; ++d )
            os << " [" << aabb[2 * d + 0] << ", " << aabb[2 * d + 1] << "],";
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
    template <typename Predicate>
    int query( Predicate const &predicates, std::vector<int> &out ) const;
};

} // end namespace DataTransferKit

#endif
