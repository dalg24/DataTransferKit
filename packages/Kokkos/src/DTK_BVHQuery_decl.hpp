#ifndef DTK_BVHQUERY_DECL_HPP
#define DTK_BVHQUERY_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include "DTK_LinearBVH.hpp"

namespace DataTransferKit
{
template <typename NO>
class BVHQuery
{
  public:
    using DeviceType = typename NO::device_type;
    static int query( BVH<NO> const bvh, Details::Nearest const &predicates,
                      Kokkos::View<int *, DeviceType> out );

    static int query( BVH<NO> const bvh, Details::Within const &predicates,
                      Kokkos::View<int *, DeviceType> out );

    // COMMENT: could also check that pointer is in the range [leaf_nodes,
    // leaf_nodes+n]
    KOKKOS_INLINE_FUNCTION
    static bool isLeaf( Node const *node )
    {
        return ( node->children_a == nullptr ) &&
               ( node->children_b == nullptr );
    }

    KOKKOS_INLINE_FUNCTION
    static int getIndex( BVH<NO> bvh, Node const *leaf )
    {
        return bvh.indices[leaf - bvh.leaf_nodes.data()];
    }

    KOKKOS_INLINE_FUNCTION
    static Node const *getRoot( BVH<NO> bvh )
    {
        return bvh.internal_nodes.data();
    }
};
}

#endif
