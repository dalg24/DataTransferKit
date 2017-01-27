#ifndef DTK_DETAILS_TREE_CONSTRUCTION_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_HPP

#include <DTK_LinearBVH.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
namespace Details
{
// utilities for tree construction
unsigned int expandBits( unsigned int v );
unsigned int morton3D( double x, double y, double z );
int countLeadingZeros( unsigned int k );
int commonPrefix( unsigned int const *k, int n, int i, int j );
int findSplit( unsigned int *sortedMortonCodes, int first, int last );
Kokkos::pair<int, int> determineRange( unsigned int *sortedMortonCodes,
                                       int numObjects, int idx );
// COMMENT: most of these could/should be protected function in BVH to avoid
// passing all this data around
void calculateBoundingBoxOfTheScene( AABB const *boundingBoxes, int n,
                                     AABB &sceneBoundingBox );
void assignMortonCodes( AABB const *boundingBoxes, unsigned int *mortonCodes,
                        int n, AABB const &sceneBoundingBox );
void sortObjects( unsigned int *morton_codes, int *object_ids, int n );
Node *generateHierarchy( unsigned int *sortedMortonCodes, int n,
                         Node *leafNodes, Node *internalNodes );
void calculateBoundingBoxes( Node *leafNodes, Node *internalNodes,
                             int numObjects, BVH *bvh );

} // end namespace Details
} // end namespace DataTransferKit

#endif
