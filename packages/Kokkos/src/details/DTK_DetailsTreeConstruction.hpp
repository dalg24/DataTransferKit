#ifndef DTK_DETAILS_TREE_CONSTRUCTION_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_HPP

#include <DTK_LinearBVH.hpp>
#include <Kokkos_Pair.hpp>

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

void sortObjects( unsigned int *morton_codes, int *object_ids, int n );
Node *generateHierarchy( unsigned int *sorted_morton_codes, int n,
                         Node *leaf_nodes, Node *internal_nodes );
void calculateBoundingBoxes( Node const *leaf_nodes, Node *internal_nodes,
                             int n );

} // end namespace Details
} // end namespace DataTransferKit

#endif
