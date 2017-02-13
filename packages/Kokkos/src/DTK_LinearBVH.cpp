#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeConstruction.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <DTK_LinearBVH.hpp>

#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{

BVH::BVH( AABB const *bounding_boxes, int n )
    : _leaf_nodes( "leaf_nodes", n )
    , _internal_nodes( "internal_nodes", n - 1 )
    , _indices( "sorted_indices", n )
{
    // determine the bounding box of the scene
    Details::calculateBoundingBoxOfTheScene( bounding_boxes, n,
                                             _internal_nodes[0].bounding_box );
    // calculate morton code of all objects
    Kokkos::View<unsigned int *, DeviceType> morton_indices( "morton", n );
    for ( int i = 0; i < n; ++i ) // parallel_for
        morton_indices[i] = Kokkos::ArithTraits<unsigned int>::max();
    Details::assignMortonCodes( bounding_boxes, morton_indices.data(), n,
                                _internal_nodes[0].bounding_box );
    // sort them along the Z-order space-filling curve
    for ( int i = 0; i < n; ++i ) // parallel_for
        _indices[i] = i;
    Details::sortObjects( morton_indices.data(), _indices.data(), n );
    // generate bounding volume hierarchy
    for ( int i = 0; i < n; ++i ) // parallel_for
        _leaf_nodes[i].bounding_box = bounding_boxes[_indices[i]];
    Details::generateHierarchy( morton_indices.data(), n, _leaf_nodes.data(),
                                _internal_nodes.data() );
    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    Details::calculateBoundingBoxes( _leaf_nodes.data(), _internal_nodes.data(),
                                     n );
}
// COMMENT: could also check that pointer is in the range [leaf_nodes,
// leaf_nodes+n]
bool BVH::isLeaf( Node const *node ) const
{
    return ( node->children.first == nullptr ) &&
           ( node->children.second == nullptr );
}
int BVH::getIndex( Node const *leaf ) const
{
    return _indices[leaf - _leaf_nodes.data()];
}
Node const *BVH::getRoot() const { return _internal_nodes.data(); }
int BVH::size() const { return _leaf_nodes.size(); }
AABB BVH::bounds() const { return getRoot()->bounding_box; }

template <typename Predicate>
int BVH::query( Predicate const &predicates,
                Kokkos::View<int *, BVH::DeviceType> out ) const
{
    using Tag = typename Predicate::Tag;
    return Details::query_dispatch( this, predicates, out, Tag{} );
}

template int BVH::query( Details::Nearest<Details::Point> const &,
                         Kokkos::View<int *, BVH::DeviceType> ) const;

template int BVH::query( Details::Within<Details::Point> const &,
                         Kokkos::View<int *, BVH::DeviceType> ) const;

} // end namespace DataTransferKit
