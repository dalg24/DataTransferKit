#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeConstruction.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <DTK_LinearBVH.hpp>

#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{

BVH::BVH( AABB const *bounding_boxes, int n )
{
    // determine the bounding box of the scene
    Details::calculateBoundingBoxOfTheScene( bounding_boxes, n,
                                             _scene_bounding_box );
    // calculate morton code of all objects
    std::vector<unsigned int> morton_indices(
        n, Kokkos::ArithTraits<unsigned int>::max() );
    Details::assignMortonCodes( bounding_boxes, morton_indices.data(), n,
                                _scene_bounding_box );
    // sort them along the Z-order space-filling curve
    _sorted_indices.resize( n, -1 );
    for ( int i = 0; i < n; ++i ) // parallel_for
        _sorted_indices[i] = i;
    Details::sortObjects( morton_indices.data(), _sorted_indices.data(), n );
    // generate bounding volume hierarchy
    _leaf_nodes.resize( n );
    for ( int i = 0; i < n; ++i ) // parallel_for
        _leaf_nodes[i].bounding_box = bounding_boxes[_sorted_indices[i]];
    _internal_nodes.resize( n - 1 );
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
int BVH::getObjectIdx( Node const *leaf_node ) const
{
    return _sorted_indices[leaf_node - _leaf_nodes.data()];
}
Node const *BVH::getRoot() const { return _internal_nodes.data(); }
Node *BVH::getRoot() { return _internal_nodes.data(); }

template <typename Predicate>
int BVH::query( Predicate const &predicates, std::vector<int> &out ) const
{
    using Tag = typename Predicate::Tag;
    return Details::query_dispatch( this, predicates, out, Tag{} );
}

template int BVH::query( Details::Nearest<Details::Point> const &,
                         std::vector<int> & ) const;

template int BVH::query( Details::Within<Details::Point> const &,
                         std::vector<int> & ) const;

} // end namespace DataTransferKit
