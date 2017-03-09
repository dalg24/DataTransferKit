#ifndef DTK_DETAILS_TREE_CONSTRUCTION_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_HPP

#include <DTK_LinearBVH.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Sort.hpp>

namespace DataTransferKit
{
namespace Details
{
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits( unsigned int v );

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D( double x, double y, double z );

namespace Functor
{
using Box = AABB;
class AssignMortonCodes
{
  public:
    AssignMortonCodes( Box const *bounding_boxes, unsigned int *morton_codes,
                       Box const &scene_bounding_box );

    void operator()( int const i ) const;

  private:
    Box const *_bounding_boxes;
    unsigned int *_morton_codes;
    Box const &_scene_bounding_box;
};

class GenerateHierarchy
{
  public:
    GenerateHierarchy( unsigned int *sorted_morton_codes, Node *leaf_nodes,
                       Node *internal_nodes, int n );

    void operator()( int const i ) const;

  private:
    unsigned int *_sorted_morton_codes;
    Node *_leaf_nodes;
    Node *_internal_nodes;
    int _n;
};
}

// utilities for tree construction
int countLeadingZeros( unsigned int k );
int commonPrefix( unsigned int const *k, int n, int i, int j );
int findSplit( unsigned int *sorted_morton_codes, int first, int last );
Kokkos::pair<int, int> determineRange( unsigned int *sorted_morton_codes, int n,
                                       int i );
// COMMENT: most of these could/should be protected function in BVH to avoid
// passing all this data around

template <typename ExecutionSpace>
void calculateBoundingBoxOfTheScene( AABB const *bounding_boxes, int n,
                                     AABB &scene_bounding_box )
{
    Functor::ExpandBoxWithBox functor( bounding_boxes );
    Kokkos::parallel_reduce( "calculate_bouding_of_the_scene",
                             Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                             functor, scene_bounding_box );
    Kokkos::fence();
}

// to assign the Morton code for a given object, we use the centroid point of
// its bounding box, and express it relative to the bounding box of the scene.
template <typename ExecutionSpace>
void assignMortonCodes( AABB const *bounding_boxes, unsigned int *morton_codes,
                        int n, AABB const &scene_bounding_box )
{
    Functor::AssignMortonCodes functor( bounding_boxes, morton_codes,
                                        scene_bounding_box );
    Kokkos::parallel_for( "assign_morton_codes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          functor );
    Kokkos::fence();
}

template <typename DeviceType>
void sortObjects( Kokkos::View<unsigned int *, DeviceType> morton_codes,
                  Kokkos::View<int *, DeviceType> object_ids, int n )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    typedef Kokkos::BinOp1D<Kokkos::View<unsigned int *, DeviceType>> CompType;

    Kokkos::Experimental::MinMaxScalar<unsigned int> result;
    Kokkos::Experimental::MinMax<unsigned int> reducer( result );
    parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        Kokkos::Impl::min_max_functor<Kokkos::View<unsigned int *, DeviceType>>(
            morton_codes ),
        reducer );
    if ( result.min_val == result.max_val )
        return;
    Kokkos::BinSort<Kokkos::View<unsigned int *, DeviceType>, CompType>
        bin_sort( morton_codes,
                  CompType( n / 2, result.min_val, result.max_val ), true );
    bin_sort.create_permute_vector();
    bin_sort.sort( morton_codes );
    bin_sort.sort( object_ids );
}

template <typename ExecutionSpace>
Node *generateHierarchy( unsigned int *sorted_morton_codes, int n,
                         Node *leaf_nodes, Node *internal_nodes )
{
    Functor::GenerateHierarchy functor( sorted_morton_codes, leaf_nodes,
                                        internal_nodes, n );
    Kokkos::parallel_for( "generate_hierarchy",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n - 1 ),
                          functor );
    Kokkos::fence();

    // Node 0 is the root.
    return &internal_nodes[0];
}

void calculateBoundingBoxes( Node const *leaf_nodes, Node *internal_nodes,
                             int n );

} // end namespace Details
} // end namespace DataTransferKit

#endif
