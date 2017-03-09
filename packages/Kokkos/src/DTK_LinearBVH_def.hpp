#ifndef DTK_LINEARBVH_DEF_HPP
#define DTK_LINEARBVH_DEF_HPP

#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeConstruction.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include "DTK_ConfigDefs.hpp"
#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{
namespace Functor
{
template <typename SC, typename LO, typename GO, typename NO>
class SetMax
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    SetMax( Kokkos::View<unsigned int *, DeviceType> morton_indices )
        : _max( Kokkos::ArithTraits<unsigned int>::max() )
        , _indices( morton_indices )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const { _indices[i] = _max; }

  private:
    unsigned int const _max;
    Kokkos::View<unsigned int *, DeviceType> _indices;
};

template <typename SC, typename LO, typename GO, typename NO>
class SetIndices
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    SetIndices( Kokkos::View<int *, DeviceType> indices )
        : _indices( indices )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const { _indices[i] = i; }

  private:
    Kokkos::View<int *, DeviceType> _indices;
};

template <typename SC, typename LO, typename GO, typename NO>
class SetBoundingBoxes
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    SetBoundingBoxes( Kokkos::View<Node *, DeviceType> leaf_nodes,
                      Kokkos::View<int *, DeviceType> indices,
                      AABB const *bounding_boxes )
        : _leaf_nodes( leaf_nodes )
        , _indices( indices )
        , _bounding_boxes( bounding_boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        _leaf_nodes[i].bounding_box = _bounding_boxes[_indices[i]];
    }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<int *, DeviceType> _indices;
    AABB const *_bounding_boxes;
};
}

template <typename SC, typename LO, typename GO, typename NO>
BVH<SC, LO, GO, NO>::BVH( AABB const *bounding_boxes, int n )
    : _leaf_nodes( "leaf_nodes", n )
    , _internal_nodes( "internal_nodes", n - 1 )
    , _indices( "sorted_indices", n )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    // determine the bounding box of the scene
    Details::calculateBoundingBoxOfTheScene<ExecutionSpace>(
        bounding_boxes, n, _internal_nodes[0].bounding_box );

    // calculate morton code of all objects
    Kokkos::View<unsigned int *, DeviceType> morton_indices( "morton", n );
    Functor::SetMax<SC, LO, GO, NO> set_max_functor( morton_indices );
    Kokkos::parallel_for( "set_morton_indices",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          set_max_functor );
    Kokkos::fence();
    Details::assignMortonCodes<ExecutionSpace>(
        bounding_boxes, morton_indices.data(), n,
        _internal_nodes[0].bounding_box );

    // sort them along the Z-order space-filling curve
    Functor::SetIndices<SC, LO, GO, NO> set_indices_functor( _indices );
    Kokkos::parallel_for( "set_indices",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          set_indices_functor );
    Kokkos::fence();
    Details::sortObjects( morton_indices, _indices, n );

    // generate bounding volume hierarchy
    Functor::SetBoundingBoxes<SC, LO, GO, NO> set_bounding_boxes_functor(
        _leaf_nodes, _indices, bounding_boxes );
    Kokkos::parallel_for( "set_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          set_bounding_boxes_functor );
    Kokkos::fence();
    Details::generateHierarchy<ExecutionSpace>(
        morton_indices.data(), n, _leaf_nodes.data(), _internal_nodes.data() );

    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    Details::calculateBoundingBoxes<ExecutionSpace>(
        _leaf_nodes.data(), _internal_nodes.data(), n );
}

// COMMENT: could also check that pointer is in the range [leaf_nodes,
// leaf_nodes+n]
template <typename SC, typename LO, typename GO, typename NO>
bool BVH<SC, LO, GO, NO>::isLeaf( Node const *node ) const
{
    return ( node->children.first == nullptr ) &&
           ( node->children.second == nullptr );
}

template <typename SC, typename LO, typename GO, typename NO>
int BVH<SC, LO, GO, NO>::getIndex( Node const *leaf ) const
{
    return _indices[leaf - _leaf_nodes.data()];
}

template <typename SC, typename LO, typename GO, typename NO>
Node const *BVH<SC, LO, GO, NO>::getRoot() const
{
    return _internal_nodes.data();
}

template <typename SC, typename LO, typename GO, typename NO>
int BVH<SC, LO, GO, NO>::size() const
{
    return _leaf_nodes.size();
}

template <typename SC, typename LO, typename GO, typename NO>
AABB BVH<SC, LO, GO, NO>::bounds() const
{
    return getRoot()->bounding_box;
}

template <typename SC, typename LO, typename GO, typename NO>
int BVH<SC, LO, GO, NO>::query(
    Details::Nearest<Details::Point> const &predicates,
    Kokkos::View<int *, typename NO::device_type> out ) const
{
    using Tag = typename Details::Nearest<Details::Point>::Tag;
    return Details::query_dispatch( this, predicates, out, Tag{} );
}

template <typename SC, typename LO, typename GO, typename NO>
int BVH<SC, LO, GO, NO>::query(
    Details::Within<Details::Point> const &predicates,
    Kokkos::View<int *, BVH::DeviceType> out ) const
{
    using Tag = typename Details::Within<Details::Point>::Tag;
    return Details::query_dispatch( this, predicates, out, Tag{} );
}

// template <typename SC, typename LO, typename GO, typename NO>
// template <typename Predicate>
// int BVH<SC,LO,GO,NO>::query( Predicate const &predicates,
//                Kokkos::View<int *, BVH::DeviceType> out ) const
//{
//    using Tag = typename Predicate::Tag;
//    return Details::query_dispatch( this, predicates, out, Tag{} );
//}
//
// template int BVH::query( Details::Nearest<Details::Point> const &,
//                         Kokkos::View<int *, BVH::DeviceType> ) const;
//
// template int BVH::query( Details::Within<Details::Point> const &,
//                         Kokkos::View<int *, BVH::DeviceType> ) const;
//
} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_LINEARBVH_INSTANT( SCALAR, LO, GO, NODE )                          \
    template class BVH<SCALAR, LO, GO, NODE>;

#endif
