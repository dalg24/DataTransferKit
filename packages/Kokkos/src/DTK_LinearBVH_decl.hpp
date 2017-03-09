#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>

#include <DTK_AABB.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_Predicate.hpp>

#include "DTK_ConfigDefs.hpp"

namespace DataTransferKit
{

struct Node
{
    virtual ~Node() = default;
    Node *parent = nullptr;
    Kokkos::pair<Node *, Node *> children = {nullptr, nullptr};
    AABB bounding_box;
};

// Bounding Volume Hierarchy
template <typename SC, typename LO, typename GO, typename NO>
class BVH
{
  public:
    using DeviceType = typename NO::device_type;

    BVH( AABB const *bounding_boxes, int n );

    int size() const;

    AABB bounds() const;

    int query( Details::Nearest<Details::Point> const &predicates,
               Kokkos::View<int *, DeviceType> out ) const;

    int query( Details::Within<Details::Point> const &predicates,
               Kokkos::View<int *, DeviceType> out ) const;

    bool isLeaf( Node const *node ) const;

    int getIndex( Node const *leaf ) const;

    Node const *getRoot() const;

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