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
// NOTE: emulate polymorphism with boolean to say whether leaf or internal node
struct Node
{
    virtual ~Node() = default;
    Node *parent = nullptr;
};
struct LeafNode : public Node
{
    LeafNode(){};
    LeafNode( int *id )
        : objectID( *id ){};
    int objectID = -1;
};
struct InternalNode : public Node
{
    InternalNode(){};
    InternalNode( Node *a, Node *b )
        : childA( a )
        , childB( b ){};
    Node *childA = nullptr;
    Node *childB = nullptr;
};
// Axis-Aligned Bounding Box
struct AABB
{
    using size_type = std::array<double, 6>::size_type;
    AABB() = default;
    AABB( std::array<double, 6> minmax )
        : _minmax( minmax )
    {
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
// Bounding Volume Hierarchy
struct BVH
{
    BVH( AABB const *boundingBoxes, int n );
    AABB getAABB( Node *node ) const;
    bool isLeaf( Node *node ) const;
    int getObjectIdx( Node *node ) const;
    Node *getLeftChild( Node *node );
    Node const *getLeftChild( Node *node ) const;
    Node *getRightChild( Node *node );
    Node const *getRightChild( Node *node ) const;
    Node *getRoot();
    Node const *getRoot() const;
    std::vector<LeafNode> _leaf_nodes;
    std::vector<InternalNode> _internal_nodes;
    std::vector<AABB> _bounding_boxes;
    AABB _scene_bounding_box; // don't actually really need to store it
};

} // end namespace DataTransferKit

#endif
