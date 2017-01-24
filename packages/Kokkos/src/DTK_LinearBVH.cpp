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
    std::vector<int> ids( n, -1 );
    iota( ids.begin(), ids.end(), 0 );
    Details::sortObjects( morton_indices.data(), ids.data(), n );
    // generate bounding volume hierarchy
    _leaf_nodes.resize( n );
    for ( auto &node : _leaf_nodes )
        node.isLeaf = true;
    _internal_nodes.resize( n - 1 );
    Details::generateHierarchy( morton_indices.data(), ids.data(), n,
                                _leaf_nodes.data(), _internal_nodes.data() );
    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    _bounding_boxes.resize( 2 * n - 1 );
    Details::calculateBoundingBoxes( _leaf_nodes.data(), _internal_nodes.data(),
                                     n, _bounding_boxes.data() );
}
AABB BVH::getAABB( Node const *node ) const
{
    int idx = -1;
    if ( isLeaf( node ) )
    {
        idx = node->objectID;
    }
    else
    {
        idx = node - _internal_nodes.data();
        idx += _leaf_nodes.size();
    }
    return _bounding_boxes[idx];
}
bool BVH::isLeaf( Node const *node ) const { return node->isLeaf; }
int BVH::getObjectIdx( Node const *leaf_node ) const
{
    return leaf_node->objectID;
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
