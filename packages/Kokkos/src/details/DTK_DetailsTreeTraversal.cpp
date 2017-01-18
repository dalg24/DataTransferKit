#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

namespace DataTransferKit
{
namespace Details
{

bool checkOverlap( AABB const &a, AABB const &b ) { return overlaps( a, b ); }

void traverseRecursive( CollisionList &list, BVH const &bvh,
                        const AABB &queryAABB, int queryObjectIdx,
                        Node const *node )
{
    // Bounding box overlaps the query => process node.
    if ( checkOverlap( bvh.getAABB( node ), queryAABB ) )
    {
        // Leaf node => report collision.
        if ( bvh.isLeaf( node ) )
            list.add( queryObjectIdx, bvh.getObjectIdx( node ) );

        // Internal node => recurse to children.
        else
        {
            Node const *childL = bvh.getLeftChild( node );
            Node const *childR = bvh.getRightChild( node );
            traverseRecursive( list, bvh, queryAABB, queryObjectIdx, childL );
            traverseRecursive( list, bvh, queryAABB, queryObjectIdx, childR );
        }
    }
}

void traverseIterative( CollisionList &list, BVH const &bvh,
                        AABB const &queryAABB, int queryObjectIdx )
{
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    Node const *stack[64];
    Node const **stackPtr = stack;
    *stackPtr++ = NULL; // push

    // Traverse nodes starting from the root.
    Node const *node = bvh.getRoot();
    do
    {
        // Check each child node for overlap.
        Node const *childL = bvh.getLeftChild( node );
        Node const *childR = bvh.getRightChild( node );
        bool overlapL = ( checkOverlap( queryAABB, bvh.getAABB( childL ) ) );
        bool overlapR = ( checkOverlap( queryAABB, bvh.getAABB( childR ) ) );

        // Query overlaps a leaf node => report collision.
        if ( overlapL && bvh.isLeaf( childL ) )
            list.add( queryObjectIdx, bvh.getObjectIdx( childL ) );

        if ( overlapR && bvh.isLeaf( childR ) )
            list.add( queryObjectIdx, bvh.getObjectIdx( childR ) );

        // Query overlaps an internal node => traverse.
        bool traverseL = ( overlapL && !bvh.isLeaf( childL ) );
        bool traverseR = ( overlapR && !bvh.isLeaf( childR ) );

        if ( !traverseL && !traverseR )
            node = *--stackPtr; // pop
        else
        {
            node = ( traverseL ) ? childL : childR;
            if ( traverseL && traverseR )
                *stackPtr++ = childR; // push
        }
    } while ( node != NULL );
}

std::list<std::pair<int, double>>
within( BVH &bvh, std::array<double, 3> const &query_point, double radius )
{
    throw std::runtime_error( "Not fully implemented." );
    std::list<std::pair<int, double>> ret;
    double x = query_point[0];
    double y = query_point[1];
    double z = query_point[2];
    AABB aabb;
    aabb._minmax = {
        x - radius, x + radius, y - radius, y + radius, z - radius, z + radius,
    };
    CollisionList collision_list;
    traverseIterative( collision_list, bvh, aabb, -1 );

    for ( auto const &collision : collision_list._ij )
    {
        int p = collision.second;
        double d; // TODO
        if ( d <= radius )
            ret.emplace_back( p, d );
    }
    ret.sort(
        []( std::pair<int, double> const &a, std::pair<int, double> const &b ) {
            return a.second > b.second;
        } );
    return ret;
}

std::list<std::pair<int, double>>
nearest( BVH &bvh, std::array<double, 3> const &query_point, int k )
{
    SortedList candidate_list( k );

    PriorityQueue queue;
    // priority does not matter for the root since the node will be processed
    // directly and removed from the priority queue
    // we don't even bother computing the distance to it
    queue.emplace( bvh.getRoot(), -1.0 );
    Node *node;
    double priority;

    double cutoff = std::numeric_limits<double>::max();
    while ( !queue.empty() && priority < cutoff )
    {
        std::tie( node, priority ) = queue.top();
        queue.pop();
        if ( bvh.isLeaf( node ) )
        {
            // QUESTION: should priority already have the correct value for the
            // distance in which case we don't need to recompute it here
            double dist = distance( query_point, bvh.getAABB( node ) );
            if ( dist < cutoff )
            {
                candidate_list.emplace( bvh.getObjectIdx( node ), dist );
                if ( candidate_list.full() )
                    cutoff = dist;
            }
        }
        else
        {
            Node *childL = bvh.getLeftChild( node );
            Node *childR = bvh.getRightChild( node );
            double distanceL = distance( query_point, bvh.getAABB( childL ) );
            double distanceR = distance( query_point, bvh.getAABB( childR ) );
            queue.emplace( childL, distanceL );
            queue.emplace( childR, distanceR );
        }
    }
    return candidate_list._sorted_list;
}

} // end namespace Details
} // end namespace DataTransferKit
