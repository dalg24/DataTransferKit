#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeConstruction.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <DTK_LinearBVH.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <functional>

namespace DataTransferKit
{

BVH::BVH( AABB const *boundingBoxes, int n )
    : _bounding_boxes( boundingBoxes, boundingBoxes + n )
{
    using std::iota;
    // determine the bounding box of the scene
    Details::calculateBoundingBoxOfTheScene( _bounding_boxes.data(), n,
                                             _scene_bounding_box );
    // calculate morton code of all objects
    std::vector<unsigned int> morton_indices(
        n, std::numeric_limits<unsigned int>::max() );
    Details::assignMortonCodes( _bounding_boxes.data(), morton_indices.data(),
                                n, _scene_bounding_box );
    // sort them along the Z-order space-filling curve
    _sorted_indices.resize( n, -1 );
    iota( _sorted_indices.begin(), _sorted_indices.end(), 0 );
    Details::sortObjects( morton_indices.data(), _sorted_indices.data(), n );
    // generate bounding volume hierarchy
    _leaf_nodes.resize( n );
    _internal_nodes.resize( n - 1 );
    Details::generateHierarchy( morton_indices.data(), n, _leaf_nodes.data(),
                                _internal_nodes.data() );
    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    _bounding_boxes.resize( 2 * n - 1 );
    Details::calculateBoundingBoxes( _leaf_nodes.data(), _internal_nodes.data(),
                                     n, this );
}
AABB &BVH::getAABB( Node const *node )
{
    int idx = -1;
    if ( isLeaf( node ) )
    {
        idx = getObjectIdx( node );
    }
    else
    {
        idx = node - _internal_nodes.data();
        idx += _leaf_nodes.size();
    }
    return _bounding_boxes[idx];
}
AABB const &BVH::getAABB( Node const *node ) const
{
    int idx = -1;
    if ( isLeaf( node ) )
    {
        idx = getObjectIdx( node );
    }
    else
    {
        idx = node - _internal_nodes.data();
        idx += _leaf_nodes.size();
    }
    return _bounding_boxes[idx];
}
// COMMENT: could also check that pointer is in the range [leaf_nodes,
// leaf_nodes+n]
bool BVH::isLeaf( Node const *node ) const
{
    return ( node->childA == nullptr ) && ( node->childB == nullptr );
}
int BVH::getObjectIdx( Node const *leaf_node ) const
{
    return _sorted_indices[leaf_node - _leaf_nodes.data()];
}
Node *BVH::getLeftChild( Node const *internal_node )
{
    return internal_node->childA;
}
Node const *BVH::getLeftChild( Node const *internal_node ) const
{
    return internal_node->childA;
}
Node *BVH::getRightChild( Node const *internal_node )
{
    return internal_node->childB;
}
Node const *BVH::getRightChild( Node const *internal_node ) const
{
    return internal_node->childB;
}
Node const *BVH::getRoot() const { return _internal_nodes.data(); }
Node *BVH::getRoot() { return _internal_nodes.data(); }

} // end namespace DataTransferKit
