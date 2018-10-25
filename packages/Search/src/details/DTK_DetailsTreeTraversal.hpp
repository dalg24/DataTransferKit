/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef DTK_DETAILS_TREE_TRAVERSAL_HPP
#define DTK_DETAILS_TREE_TRAVERSAL_HPP

#include <DTK_DBC.hpp>

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsPriorityQueue.hpp>
#include <DTK_DetailsStack.hpp>
#include <DTK_Predicates.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BoundingVolumeHierarchy;

namespace Details
{
template <typename DeviceType>
struct TreeTraversal
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    template <typename Predicate, typename... Args>
    KOKKOS_INLINE_FUNCTION static int
    query( BoundingVolumeHierarchy<DeviceType> const &bvh,
           Predicate const &pred, Args &&... args )
    {
        using Tag = typename Predicate::Tag;
        return queryDispatch( Tag{}, bvh, pred, std::forward<Args>( args )... );
    }

    /**
     * Return true if the node is a leaf.
     */
    KOKKOS_INLINE_FUNCTION
    static bool isLeaf( Node const *node )
    {
        return ( node->children.first == nullptr );
    }

    /**
     * Return the index of the leaf node.
     */
    KOKKOS_INLINE_FUNCTION
    static size_t getIndex( Node const *leaf )
    {
        static_assert( sizeof( size_t ) == sizeof( Node * ),
                       "Conversion is a bad idea if these sizes do not match" );
        return reinterpret_cast<size_t>( leaf->children.second );
    }

    // DO NOT USE IN PRODUCTION CODE THIS IS FOR VISUALIZATION PURPOSES ONLY
    KOKKOS_INLINE_FUNCTION
    static Node const *getLeaf( BoundingVolumeHierarchy<DeviceType> const &bvh,
                                size_t index )
    {
        Node const *leaf = ( bvh._leaf_nodes ).data();
        for ( int i = 0; i < ( bvh._leaf_nodes ).extent_int( 0 ); ++i, ++leaf )
            if ( index == getIndex( leaf ) )
                return leaf;
        return nullptr;
    }

    /**
     * Return the root node of the BVH.
     */
    KOKKOS_INLINE_FUNCTION
    static Node const *getRoot( BoundingVolumeHierarchy<DeviceType> const &bvh )
    {
        if ( bvh.empty() )
            return nullptr;
        return ( bvh.size() > 1 ? bvh._internal_nodes : bvh._leaf_nodes )
            .data();
    }

    KOKKOS_INLINE_FUNCTION
    static size_t getIndex( BoundingVolumeHierarchy<DeviceType> const &bvh,
                            Node const *node )
    {
        if ( isLeaf( node ) )
            return getIndex( node );
        else
            return node - getRoot( bvh );
    }
};
template <typename DeviceType>
std::string getNodeLabel( BoundingVolumeHierarchy<DeviceType> const &bvh,
                          Node const *node )
{
    auto const node_is_leaf = TreeTraversal<DeviceType>::isLeaf( node );
    auto const node_index = TreeTraversal<DeviceType>::getIndex( bvh, node );
    std::string label = node_is_leaf ? "l" : "i";
    label.append( std::to_string( node_index ) );
    return label;
}

template <typename DeviceType>
std::string getNodeAttributes( BoundingVolumeHierarchy<DeviceType> const &bvh,
                               Node const *node )
{
    auto const node_is_leaf = TreeTraversal<DeviceType>::isLeaf( node );
    std::string attributes = node_is_leaf ? "[leaf]" : "[internal]";
    return attributes;
}

template <typename DeviceType>
std::string getEdgeAttributes( BoundingVolumeHierarchy<DeviceType> const &bvh,
                               Node const *parent, Node const *child )
{
    auto const child_is_leaf = TreeTraversal<DeviceType>::isLeaf( child );
    std::string attributes = child_is_leaf ? "[pendant]" : "[edge]";
    return attributes;
}

// traverse is recursive
template <typename DeviceType>
void printTikZ( BoundingVolumeHierarchy<DeviceType> const &bvh,
                Node const *node, std::ostream &os )
{
    auto const node_label = getNodeLabel( bvh, node );
    auto const node_attributes = getNodeAttributes( bvh, node );
    auto const node_is_internal = !TreeTraversal<DeviceType>::isLeaf( node );
    os << " child{ node ";
    os << "[" << node_attributes << "] ";
    os << "(" << node_label << ") {" << node_label << "}";
    if ( node_is_internal )
        for ( Node const *child :
              {node->children.first, node->children.second} )
            printTikZ( bvh, child, os );
    os << " }";
}

struct GraphvizVisitor
{
    template <typename DeviceType>
    static void visit( BoundingVolumeHierarchy<DeviceType> const &bvh,
                       Node const *node, std::ostream &os )
    {
        visitNode( bvh, node, os );
        visitEdgesStartingFromNode( bvh, node, os );
    }

    template <typename DeviceType>
    static void visit( BoundingVolumeHierarchy<DeviceType> const &bvh,
                       Node const *leaf, double distance, std::ostream &os )
    {
        auto const leaf_label = getNodeLabel( bvh, leaf );
        std::string const leaf_attributes = "[result]";
        std::string const commented_line = "// distance from " + leaf_label +
                                           " is " + std::to_string( distance );
        os << "    " << leaf_label << " " << leaf_attributes << ";\n";
        os << "    " << commented_line << "\n";
    }

    template <typename DeviceType>
    static void visitNode( BoundingVolumeHierarchy<DeviceType> const &bvh,
                           Node const *node, std::ostream &os )
    {
        auto const node_label = getNodeLabel( bvh, node );
        auto const node_attributes = getNodeAttributes( bvh, node );

        os << "    " << node_label << " " << node_attributes << ";\n";
    }

    template <typename DeviceType>
    static void
    visitEdgesStartingFromNode( BoundingVolumeHierarchy<DeviceType> const &bvh,
                                Node const *node, std::ostream &os )
    {
        auto const node_label = getNodeLabel( bvh, node );
        auto const node_is_internal =
            !TreeTraversal<DeviceType>::isLeaf( node );

        if ( node_is_internal )
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                auto const child_label = getNodeLabel( bvh, child );
                auto const edge_attributes =
                    getEdgeAttributes( bvh, node, child );

                os << "    " << node_label << " -> " << child_label << " "
                   << edge_attributes << ";\n";
            }
    }
};

// std::ostream &operator<<( std::ostream &os, Point const &p )
// {
//   os << "(" << p[0] << "," << p[1] << ")";
//   return os;
// }

struct BoundingVolumesVisitor
{
    static void printPoint( Point const &p, std::ostream &os )
    {
        os << "(" << p[0] << "," << p[1] << ")";
    }

    template <typename DeviceType>
    static void visit( BoundingVolumeHierarchy<DeviceType> const &bvh,
                       Node const *node, std::ostream &os )
    {
        auto const node_label = getNodeLabel( bvh, node );
        auto const node_attributes = getNodeAttributes( bvh, node );
        auto const bounding_volume = node->bounding_box;
        auto const min_corner = bounding_volume.minCorner();
        auto const max_corner = bounding_volume.maxCorner();
        bool const node_is_leaf = TreeTraversal<DeviceType>::isLeaf( node );
        os << R"(\draw)" << node_attributes << " ";
        printPoint( min_corner, os );
        os << " rectangle ";
        printPoint( max_corner, os );
        os << " node {" << node_label << "}";
        os << ";\n";
    }

    template <typename DeviceType>
    static void visit( BoundingVolumeHierarchy<DeviceType> const &bvh,
                       Node const *leaf, double distance, std::ostream &os )
    {
        std::stringstream ss;
        visit( bvh, leaf, ss );
        std::string tmp = ss.str();
        std::string const old_tag = "leaf";
        std::string const new_tag = "result";
        tmp.replace( tmp.find( old_tag ), old_tag.length(), new_tag );
        os << tmp;
    }
};

template <typename Visitor, typename DeviceType>
void visitAllIterative( BoundingVolumeHierarchy<DeviceType> const &bvh,
                        std::ostream &os )
{
    Stack<Node const *> stack;
    stack.emplace( TreeTraversal<DeviceType>::getRoot( bvh ) );
    while ( !stack.empty() )
    {
        Node const *node = stack.top();
        stack.pop();

        Visitor::visit( bvh, node, os );

        if ( !TreeTraversal<DeviceType>::isLeaf( node ) )
            for ( Node const *child :
                  {node->children.first, node->children.second} )
                stack.push( child );
    }
}

template <typename DeviceType>
void visitAllRecursive( BoundingVolumeHierarchy<DeviceType> const &bvh,
                        std::ostream &os )
{
    os << "\\node [internal] (i0) {i0}";
    Node const *root = TreeTraversal<DeviceType>::getRoot( bvh );
    for ( Node const *child : {root->children.first, root->children.second} )
        printTikZ( bvh, child, os );
    os << ";\n";
}

// There are two (related) families of search: one using a spatial predicate and
// one using nearest neighbours query (see boost::geometry::queries
// documentation).
template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_FUNCTION int
spatialQuery( BoundingVolumeHierarchy<DeviceType> const &bvh,
              Predicate const &predicate, Insert const &insert )
{
    if ( bvh.empty() )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        if ( predicate( leaf ) )
        {
            int const leaf_index = TreeTraversal<DeviceType>::getIndex( leaf );
            insert( leaf_index );
            return 1;
        }
        else
            return 0;
    }

    Stack<Node const *> stack;

    stack.emplace( TreeTraversal<DeviceType>::getRoot( bvh ) );
    int count = 0;

    while ( !stack.empty() )
    {
        Node const *node = stack.top();
        stack.pop();

        if ( TreeTraversal<DeviceType>::isLeaf( node ) )
        {
            insert( TreeTraversal<DeviceType>::getIndex( node ) );
            count++;
        }
        else
        {
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                if ( predicate( child ) )
                {
                    stack.push( child );
                }
            }
        }
    }
    return count;
}

// query k nearest neighbours
template <typename DeviceType, typename Distance, typename Insert,
          typename Buffer>
KOKKOS_FUNCTION int
nearestQuery( BoundingVolumeHierarchy<DeviceType> const &bvh,
              Distance const &distance, std::size_t k, Insert const &insert,
              Buffer const &buffer )
{
    if ( bvh.empty() || k < 1 )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        int const leaf_index = TreeTraversal<DeviceType>::getIndex( leaf );
        double const leaf_distance = distance( leaf );
        insert( leaf_index, leaf_distance );
        return 1;
    }

    // Nodes with a distance that exceed that radius can safely be discarded.
    // Initialize the radius to infinity and tighten it once k neighbors have
    // been found.
    double radius = KokkosHelpers::ArithTraits<double>::infinity();

    using PairIndexDistance = Kokkos::pair<int, double>;
    static_assert(
        std::is_same<typename Buffer::value_type, PairIndexDistance>::value,
        "Type of the elements stored in the buffer passed as argument to "
        "TreeTraversal::nearestQuery is not right" );
    struct CompareDistance
    {
        KOKKOS_INLINE_FUNCTION bool
        operator()( PairIndexDistance const &lhs,
                    PairIndexDistance const &rhs ) const
        {
            return lhs.second < rhs.second;
        }
    };
    // Use a priority queue for convenience to store the results and preserve
    // the heap structure internally at all time.  There is no memory
    // allocation, elements are stored in the buffer passed as an argument.
    // The farthest leaf node is on top.
    assert( k == buffer.size() );
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap( UnmanagedStaticVector<PairIndexDistance>( buffer.data(),
                                                        buffer.size() ) );

    using PairNodePtrDistance = Kokkos::pair<Node const *, double>;
    Stack<PairNodePtrDistance> stack;
    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    stack.emplace( TreeTraversal<DeviceType>::getRoot( bvh ), 0. );

    while ( !stack.empty() )
    {
        Node const *node = stack.top().first;
        double const node_distance = stack.top().second;
        stack.pop();

        if ( node_distance < radius )
        {
            if ( TreeTraversal<DeviceType>::isLeaf( node ) )
            {
                int const leaf_index =
                    TreeTraversal<DeviceType>::getIndex( node );
                double const leaf_distance = node_distance;
                if ( heap.size() < k )
                {
                    // Insert leaf node and update radius if it was the kth one.
                    heap.push( Kokkos::make_pair( leaf_index, leaf_distance ) );
                    if ( heap.size() == k )
                        radius = heap.top().second;
                }
                else
                {
                    // Replace top element in the heap and update radius.
                    heap.popPush(
                        Kokkos::make_pair( leaf_index, leaf_distance ) );
                    radius = heap.top().second;
                }
            }
            else
            {
                // Insert children into the stack and make sure that the
                // closest one ends on top.
                Node const *left_child = node->children.first;
                double const left_child_distance = distance( left_child );
                Node const *right_child = node->children.second;
                double const right_child_distance = distance( right_child );
                if ( left_child_distance < right_child_distance )
                {
                    // NOTE not really sure why but it performed better with
                    // the conditional insertion on the device and without it
                    // on the host (~5% improvement for both)
#if defined( __CUDA_ARCH__ )
                    if ( right_child_distance < radius )
#endif
                        stack.emplace( right_child, right_child_distance );
                    stack.emplace( left_child, left_child_distance );
                }
                else
                {
#if defined( __CUDA_ARCH__ )
                    if ( left_child_distance < radius )
#endif
                        stack.emplace( left_child, left_child_distance );
                    stack.emplace( right_child, right_child_distance );
                }
            }
        }
    }
    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap( heap.data(), heap.data() + heap.size(), heap.valueComp() );
    for ( decltype( heap.size() ) i = 0; i < heap.size(); ++i )
    {
        int const leaf_index = ( heap.data() + i )->first;
        double const leaf_distance = ( heap.data() + i )->second;
        insert( leaf_index, leaf_distance );
    }
    return heap.size();
}

template <typename DeviceType, typename Distance, typename Insert>
KOKKOS_FUNCTION int
oldNearestQuery( BoundingVolumeHierarchy<DeviceType> const &bvh,
                 Distance const &distance, std::size_t k, Insert const &insert )
{
    if ( bvh.empty() || k < 1 )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        int const leaf_index = TreeTraversal<DeviceType>::getIndex( leaf );
        double const leaf_distance = distance( leaf );
        insert( leaf_index, leaf_distance );
        return 1;
    }

    using PairNodePtrDistance = Kokkos::pair<Node const *, double>;
    struct CompareDistance
    {
        KOKKOS_INLINE_FUNCTION bool
        operator()( PairNodePtrDistance const &lhs,
                    PairNodePtrDistance const &rhs ) const
        {
            // reverse order (larger distance means lower priority)
            return lhs.second > rhs.second;
        }
    };
    PriorityQueue<PairNodePtrDistance, CompareDistance,
                  StaticVector<PairNodePtrDistance, 256>>
        queue;

    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    queue.emplace( TreeTraversal<DeviceType>::getRoot( bvh ), 0. );
    std::size_t count = 0;

    while ( !queue.empty() && count < k )
    {
        // get the node that is on top of the priority list (i.e. is the
        // closest to the query point)
        Node const *node = queue.top().first;
        double const node_distance = queue.top().second;

        if ( TreeTraversal<DeviceType>::isLeaf( node ) )
        {
            queue.pop();
            int const leaf_index = TreeTraversal<DeviceType>::getIndex( node );
            double const leaf_distance = node_distance;
            insert( leaf_index, leaf_distance );
            ++count;
        }
        else
        {
            // Insert children into the priority queue
            Node const *left_child = node->children.first;
            double const left_child_distance = distance( left_child );
            Node const *right_child = node->children.second;
            double const right_child_distance = distance( right_child );
            queue.popPush( left_child, left_child_distance );
            queue.emplace( right_child, right_child_distance );
        }
    }
    return count;
}

template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_INLINE_FUNCTION int
queryDispatch( SpatialPredicateTag,
               BoundingVolumeHierarchy<DeviceType> const &bvh,
               Predicate const &pred, Insert const &insert )
{
    return spatialQuery( bvh, pred, insert );
}

template <typename DeviceType, typename Predicate, typename Insert,
          typename Buffer>
KOKKOS_INLINE_FUNCTION int queryDispatch(
    NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
    Predicate const &pred, Insert const &insert, Buffer const &buffer )
{
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    return nearestQuery( bvh,
                         [geometry]( Node const *node ) {
                             return distance( geometry, node->bounding_box );
                         },
                         k, insert, buffer );
}

template <typename DeviceType, typename Predicate,
          typename Visitor = GraphvizVisitor>
KOKKOS_INLINE_FUNCTION int
visit( BoundingVolumeHierarchy<DeviceType> const &bvh, Predicate const &pred,
       std::ostream &os )
{
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    Kokkos::View<Kokkos::pair<int, double> *, DeviceType> buffer( "buffer", k );
    int const count =
        nearestQuery( bvh,
                      [geometry, &os, &bvh]( Node const *node ) {
                          Visitor::visit( bvh, node, os );
                          return distance( geometry, node->bounding_box );
                      },
                      k,
                      [&os, &bvh]( int index, double distance ) {
                          Node const *leaf =
                              TreeTraversal<DeviceType>::getLeaf( bvh, index );
                          Visitor::visit( bvh, leaf, distance, os );
                      },
                      buffer );
    return count;
}

template <typename DeviceType, typename Predicate,
          typename Visitor = GraphvizVisitor>
KOKKOS_INLINE_FUNCTION int
visitOld( BoundingVolumeHierarchy<DeviceType> const &bvh, Predicate const &pred,
          std::ostream &os )
{
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    int const count = oldNearestQuery(
        bvh,
        [geometry, &os, &bvh]( Node const *node ) {
            Visitor::visit( bvh, node, os );
            return distance( geometry, node->bounding_box );
        },
        k,
        [&os, &bvh]( int index, double distance ) {
            Node const *leaf = TreeTraversal<DeviceType>::getLeaf( bvh, index );
            Visitor::visit( bvh, leaf, distance, os );
        } );
    return count;
}

} // namespace Details
} // namespace DataTransferKit

#endif
