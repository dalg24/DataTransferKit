#ifndef DTK_DETAILS_TREE_TRAVERSAL_HPP
#define DTK_DETAILS_TREE_TRAVERSAL_HPP

#include <details/DTK_DetailsAlgorithms.hpp> // overlap TODO:remove it
#include <details/DTK_Predicate.hpp>

#include <DTK_LinearBVH.hpp> // BVH

#include <functional>
#include <list>
#include <queue>
#include <stack>
#include <utility> // std::pair, std::make_pair

namespace DataTransferKit
{

struct Node;

namespace Details
{

struct CollisionList
{
    void add( int i, int j ) { _ij.emplace_back( i, j ); }
    std::vector<std::pair<int, int>> _ij;
};

template <typename NO>
void traverseRecursive( CollisionList &list,
                        ::DataTransferKit::BVH<NO> const &bvh,
                        AABB const &queryAABB, int queryObjectIdx,
                        Node const *node );

template <typename NO>
void traverseIterative( CollisionList &list, BVH<NO> const &bvh,
                        AABB const &queryAABB, int queryObjectIdx );
// TODO: get rid of this guy
bool checkOverlap( AABB const &a, AABB const &b );

// query k nearest neighbours
template <typename NO>
std::list<std::pair<int, double>>
nearest( BVH<NO> const &bvh, Point const &query_point, int k = 1 );

// radius search
template <typename NO>
std::list<std::pair<int, double>>
within( BVH<NO> const &bvh, Point const &query_point, double radius );

// priority queue helper for nearest neighbor search
using Value = std::pair<Node const *, double>;
struct CompareDistance
{
    bool operator()( Value const &lhs, Value const &rhs )
    {
        // larger distance means lower priority
        return lhs.second > rhs.second;
    }
};
using PriorityQueue =
    std::priority_queue<Value, std::vector<Value>, CompareDistance>;
// helper for k nearest neighbors search
// container to store candidates throughout the search
struct SortedList
{
    using Value = std::pair<int, double>;
    using Container = std::list<Value>;
    SortedList( int k )
        : _maxsize( k ){};
    template <typename... Args>
    void emplace( Args &&... args )
    {
        if ( _sorted_list.size() < _maxsize )
            _sorted_list.emplace_back( std::forward<Args>( args )... );
        else
        {
            Value val( std::forward<Args>( args )... );
            if ( !_compare( val, _sorted_list.back() ) )
                throw std::runtime_error(
                    "Error in DTK::Impl::SortedList::emplace()" );
            _sorted_list.back() = val;
        }
        _sorted_list.sort( _compare );
    }
    bool empty() const { return _sorted_list.empty(); }
    bool full() const { return _sorted_list.size() == _maxsize; }
    Container::size_type size() const { return _sorted_list.size(); }
    Value const &back() const { return _sorted_list.back(); }
    Container _sorted_list;
    // closer objects stored at the front of the list
    std::function<bool( Value const &a, Value const &b )> _compare =
        []( Value const &a, Value const &b ) { return a.second < b.second; };
    Container::size_type _maxsize;
};
using Stack = std::stack<Value, std::vector<Value>>;

template <typename NO, typename Predicate>
int query_dispatch( BVH<NO> const *bvh, Predicate const &pred,
                    Kokkos::View<int *, typename NO::device_type> out,
                    NearestPredicateTag )
{
    auto nearest_neighbors = nearest( *bvh, pred._geometry, pred._k );
    int const n = nearest_neighbors.size();
    out = Kokkos::View<int *, typename NO::device_type>( "dummy", n );
    int i = 0;
    for ( auto const &elem : nearest_neighbors )
    {
        out[i++] = elem.first;
    }
    if ( i != n )
        throw std::runtime_error( "ohhh no" );
    return n;
}

template <typename NO, typename Predicate>
int query_dispatch( BVH<NO> const *bvh, Predicate const &pred,
                    Kokkos::View<int *, typename NO::device_type> out,
                    SpatialPredicateTag )
{
    auto aaa = within( *bvh, pred._geometry, pred._radius );
    int const n = aaa.size();
    out = Kokkos::View<int *, typename NO::device_type>( "dummy", n );
    int i = 0;
    for ( auto const &elem : aaa )
    {
        out[i++] = elem.first;
    }
    return n;
}

template <typename NO>
void traverseRecursive( CollisionList &list, BVH<NO> const &bvh,
                        const AABB &queryAABB, int queryObjectIdx,
                        Node const *node )
{
    // Bounding box overlaps the query => process node.
    if ( checkOverlap( node->bounding_box, queryAABB ) )
    {
        // Leaf node => report collision.
        if ( bvh.isLeaf( node ) )
            list.add( queryObjectIdx, bvh.getIndex( node ) );

        // Internal node => recurse to children.
        else
        {
            Node const *childL = node->children.first;
            Node const *childR = node->children.second;
            traverseRecursive( list, bvh, queryAABB, queryObjectIdx, childL );
            traverseRecursive( list, bvh, queryAABB, queryObjectIdx, childR );
        }
    }
}

template <typename NO>
void traverseIterative( CollisionList &list, BVH<NO> const &bvh,
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
        Node const *childL = node->children.first;
        Node const *childR = node->children.second;
        bool overlapL = ( checkOverlap( queryAABB, childL->bounding_box ) );
        bool overlapR = ( checkOverlap( queryAABB, childR->bounding_box ) );

        // Query overlaps a leaf node => report collision.
        if ( overlapL && bvh.isLeaf( childL ) )
            list.add( queryObjectIdx, bvh.getIndex( childL ) );

        if ( overlapR && bvh.isLeaf( childR ) )
            list.add( queryObjectIdx, bvh.getIndex( childR ) );

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

template <typename NO>
std::list<std::pair<int, double>>
within( BVH<NO> const &bvh, Point const &query_point, double radius )
{
    std::list<std::pair<int, double>> ret;

    Stack stack;

    Node const *node = bvh.getRoot();
    double node_distance = 0.0;
    stack.emplace( node, node_distance );
    while ( !stack.empty() )
    {
        std::tie( node, node_distance ) = stack.top();
        stack.pop();
        if ( bvh.isLeaf( node ) )
        {
            ret.emplace_back( bvh.getIndex( node ), node_distance );
        }
        else
        {
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                double child_distance =
                    distance( query_point, child->bounding_box );
                if ( child_distance <= radius )
                    stack.emplace( child, child_distance );
            }
        }
    }
    ret.sort(
        []( std::pair<int, double> const &a, std::pair<int, double> const &b ) {
            return a.second > b.second;
        } );
    return ret;
}

template <typename NO>
std::list<std::pair<int, double>> nearest( BVH<NO> const &bvh,
                                           Point const &query_point, int k )
{
    SortedList candidate_list( k );

    PriorityQueue queue;
    // priority does not matter for the root since the node will be processed
    // directly and removed from the priority queue
    // we don't even bother computing the distance to it
    Node const *node = bvh.getRoot();
    double node_distance = 0.0;
    queue.emplace( node, node_distance );

    double cutoff = Kokkos::ArithTraits<double>::max();
    while ( !queue.empty() && node_distance < cutoff )
    {
        // get the node that is on top of the priority list (i.e. is the closest
        // to the query point)
        std::tie( node, node_distance ) = queue.top();
        queue.pop();
        if ( bvh.isLeaf( node ) )
        {
            if ( node_distance < cutoff )
            {
                // add leaf node to the candidate list
                candidate_list.emplace( bvh.getIndex( node ), node_distance );
                // update cutoff if k neighbors are in the list
                if ( candidate_list.full() )
                    std::tie( std::ignore, cutoff ) = candidate_list.back();
            }
        }
        else
        {
            // insert children of the node in the priority list
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                double child_distance =
                    distance( query_point, child->bounding_box );
                queue.emplace( child, child_distance );
            }
        }
    }
    return candidate_list._sorted_list;
}

} // end namespace Details
} // end namespace DataTransferKit

#endif
