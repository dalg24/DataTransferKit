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
unsigned int morton3D( float x, float y, float z );
int findSplit( unsigned int *sortedMortonCodes, int first, int last );

Kokkos::pair<int, int> determineRange( unsigned int *sortedMortonCodes,
                                       int numObjects, int idx );
// COMMENT: most of these could/should be protected function in BVH to avoid
// passing all this data around
void sortObjects( unsigned int *mortonCodes, int *objectIDs, int n );
void assignMortonCodes( AABB const *boundingBoxes, unsigned int *mortonCodes,
                        int n, AABB const &sceneBoundingBox );
void calculateBoundingBoxOfTheScene( AABB const *boundingBoxes, int n,
                                     AABB &sceneBoundingBox );
// fully parallel version
Node *generateHierarchy( unsigned int *sortedMortonCodes, int *sortedObjectIDs,
                         int n, LeafNode *leafNodes,
                         InternalNode *internalNodes );
// less efficient top-bottom version
Node *generateHierarchy( unsigned int *sortedMortonCodes, int *sortedObjectIDs,
                         int first, int last );
void calculateBoundingBoxes( LeafNode *leafNodes, InternalNode *internalNodes,
                             int numObjects, AABB *boundingBoxes );

} // end namespace Details
} // end namespace DataTransferKit

#endif
