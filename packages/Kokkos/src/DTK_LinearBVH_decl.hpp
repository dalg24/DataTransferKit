#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>

#include <DTK_Box.hpp>
#include <DTK_Node.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_Predicate.hpp>

#include "DTK_ConfigDefs.hpp"

namespace DataTransferKit
{

// Bounding Volume Hierarchy
template <typename NO>
class BVH
{
  public:
    using DeviceType = typename NO::device_type;

    BVH();

    BVH( Kokkos::View<BBox *, DeviceType> bounding_boxes );

    int size() const;

    BBox bounds() const;

    int query( Details::Nearest const &predicates,
               Kokkos::View<int *, DeviceType> out ) const;

    int query( Details::Within const &predicates,
               Kokkos::View<int *, DeviceType> out ) const;

    // COMMENT: could also check that pointer is in the range [leaf_nodes,
    // leaf_nodes+n]
    KOKKOS_INLINE_FUNCTION
    bool isLeaf( Node const *node ) const
    {
        return ( node->children_a == nullptr ) &&
               ( node->children_b == nullptr );
    }

    KOKKOS_INLINE_FUNCTION
    int getIndex( Node const *leaf ) const
    {
        return _indices[leaf - _leaf_nodes.data()];
    }

    KOKKOS_INLINE_FUNCTION
    Node const *getRoot() const { return _internal_nodes.data(); }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
    // Array of indices that sort the boxes used to construct the hierarchy.
    // The leaf nodes are ordered so we need these to identify objects that meet
    // a predicate.
    Kokkos::View<int *, DeviceType> _indices;
};

} // end namespace DataTransferKit

#endif
