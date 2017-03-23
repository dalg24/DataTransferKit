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

// There are two (related) families of search: one using a spatial predicate and
// one using nearest neighbours query (see boost::geometry::queries
// documentation).
template <typename NO, typename Predicate>
std::list<int> spatial_query( BVH<NO> const &bvh, Point const &query_point,
                              Predicate const &predicate );

// query k nearest neighbours
template <typename NO>
std::list<std::pair<int, double>>
nearest_query( BVH<NO> const &bvh, Point const &query_point, int k = 1 );

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

using Stack = std::stack<Node const *, std::vector<Node const *>>;

template <typename NO, typename Predicate>
int query_dispatch( BVH<NO> const *bvh, Predicate const &pred,
                    Kokkos::View<int *, typename NO::device_type> out,
                    NearestPredicateTag )
{
    auto nearest_neighbors = nearest_query( *bvh, pred._geometry, pred._k );
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
    auto aaa = spatial_query( *bvh, pred );
    int const n = aaa.size();
    out = Kokkos::View<int *, typename NO::device_type>( "dummy", n );
    int i = 0;
    for ( auto const &elem : aaa )
    {
        out[i++] = elem;
    }
    return n;
}

template <typename NO, typename Predicate>
std::list<int> spatial_query( BVH<NO> const &bvh, Predicate const &predicate )
{
    std::list<int> ret;

    Stack stack;

    Node const *node = bvh.getRoot();
    stack.emplace( node );
    while ( !stack.empty() )
    {
        node = stack.top();
        stack.pop();
        if ( bvh.isLeaf( node ) )
        {
            ret.emplace_back( bvh.getIndex( node ) );
        }
        else
        {
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                if ( predicate( child ) )
                    stack.emplace( child );
            }
        }
    }

    return ret;
}

template <typename NO>
std::list<std::pair<int, double>>
nearest_query( BVH<NO> const &bvh, Point const &query_point, int k )
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
