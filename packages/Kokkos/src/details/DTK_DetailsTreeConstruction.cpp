#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeConstruction.hpp>

#include <algorithm>
#include <atomic>
#include <cassert> // TODO: probably want to use something else but fine for now

namespace DataTransferKit
{
namespace Details
{

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits( unsigned int v )
{
    v = ( v * 0x00010001u ) & 0xFF0000FFu;
    v = ( v * 0x00000101u ) & 0x0F00F00Fu;
    v = ( v * 0x00000011u ) & 0xC30C30C3u;
    v = ( v * 0x00000005u ) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D( float x, float y, float z )
{
    using std::min;
    using std::max;
    x = min( max( x * 1024.0f, 0.0f ), 1023.0f );
    y = min( max( y * 1024.0f, 0.0f ), 1023.0f );
    z = min( max( z * 1024.0f, 0.0f ), 1023.0f );
    unsigned int xx = expandBits( (unsigned int)x );
    unsigned int yy = expandBits( (unsigned int)y );
    unsigned int zz = expandBits( (unsigned int)z );
    return xx * 4 + yy * 2 + zz;
}

#define __clz( x ) __builtin_clz( x )

int findSplit( unsigned int *sortedMortonCodes, int first, int last )
{
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if ( firstCode == lastCode )
        return ( first + last ) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz( firstCode ^ lastCode );

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = ( step + 1 ) >> 1;    // exponential decrease
        int newSplit = split + step; // proposed new position

        if ( newSplit < last )
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz( firstCode ^ splitCode );
            if ( splitPrefix > commonPrefix )
                split = newSplit; // accept proposal
        }
    } while ( step > 1 );

    return split;
}

// branchless sign function
// QUESTION: do we want to add it to DTK helpers?
inline int sgn( int x ) { return ( x > 0 ) - ( x < 0 ); }

// COMMENT: Do I want to expose this in the headers?
// TODO: Answer is YES.  And we need tests for that
// More particularly need to handle the case when i!=j but codes are the same
int commonPrefix( unsigned int *mortonCodes, int numObjects, int i, int j )
{
    // int __builtin_clz(unsigned int x) result is undefined if x is 0
    if ( i == j )
        return std::numeric_limits<unsigned int>::digits;
    for ( int k : {i, j} )
        if ( k < 0 || k > numObjects - 1 )
            return -1;
    return __clz( mortonCodes[i] ^ mortonCodes[j] );
}

Kokkos::pair<int, int> determineRange( unsigned int *sortedMortonCodes,
                                       int numObjects, int idx )
{
    using std::min;
    using std::max;
    // determine direction of the range (+1 or -1)
    int direction =
        sgn( commonPrefix( sortedMortonCodes, numObjects, idx, idx + 1 ) -
             commonPrefix( sortedMortonCodes, numObjects, idx, idx - 1 ) );
    assert( direction == +1 || direction == -1 );
    // compute upper bound for the length of the range
    int upperBoundLenght = 2;
    int commonPrefixLowerBound =
        commonPrefix( sortedMortonCodes, numObjects, idx, idx - direction );
    // compute upper bound for the length of the range
    while ( commonPrefix( sortedMortonCodes, numObjects, idx,
                          idx + direction * upperBoundLenght ) >
            commonPrefixLowerBound )
    {
        upperBoundLenght = upperBoundLenght << 1;
    }
    // find the other end using binary search
    int split = 0;
    int step = upperBoundLenght;
    do
    {
        step = step >> 1;
        int newSplit = idx + ( split + step ) * direction;
        //    unsigned int splitCode = sortedMortonCodes[newSplit];
        if ( commonPrefix( sortedMortonCodes, numObjects, idx, newSplit ) >
             commonPrefixLowerBound )
            split += step;
    } while ( step > 1 );
    //  std::cout<<idx<<"  dir="<<direction<<"
    //  delta_min="<<commonPrefixLowerBound<<"  lmax="<<upperBoundLenght<<"
    //  length="<<split<<"\n";
    int jdx = idx + split * direction;
    // an equivalent to std::make_pair or std::minmax would be nice here
    return Kokkos::pair<int, int>( min( idx, jdx ), max( idx, jdx ) );
}

// alternate top-down implementation less efficient
Node *generateHierarchy( unsigned int *sortedMortonCodes, int *sortedObjectIDs,
                         int first, int last )
{
    // Single object => create a leaf node.

    if ( first == last )
        return new LeafNode( &sortedObjectIDs[first] );

    // Determine where to split the range.

    int split = findSplit( sortedMortonCodes, first, last );

    // Process the resulting sub-ranges recursively.

    Node *childA =
        generateHierarchy( sortedMortonCodes, sortedObjectIDs, first, split );
    Node *childB = generateHierarchy( sortedMortonCodes, sortedObjectIDs,
                                      split + 1, last );
    return new InternalNode( childA, childB );
}

void sortObjects( unsigned int *mortonCodes, int *objectIDs, int n )
{
    using std::sort;
    // possibly use thrust::sort()
    // see https://thrust.github.io
    sort( objectIDs, objectIDs + n,
          [&mortonCodes]( int const &i, int const &j ) {
              return mortonCodes[i] < mortonCodes[j];
          } );
    // TODO: in-place permutation of mortonCodes rather than 2nd sort
    sort( mortonCodes, mortonCodes + n );
}

// to assign the Morton code for a given object, we use the centroid point of
// its bounding box, and express it relative to the bounding box of the scene.
void assignMortonCodes( AABB const *boundingBoxes, unsigned int *mortonCodes,
                        int n, AABB const &sceneBoundingBox )
{
    std::array<double, 3> xyz;
    double a, b;
    for ( int i = 0; i < n; ++i ) // parallel for
    {
        for ( int d = 0; d < 3; ++d )
        {
            xyz[d] = 0.5 * ( boundingBoxes[i]._minmax[2 * d + 0] +
                             boundingBoxes[i]._minmax[2 * d + 1] );
            a = sceneBoundingBox._minmax[2 * d + 0];
            b = sceneBoundingBox._minmax[2 * d + 1];
            xyz[d] = ( xyz[d] - a ) / ( b - a );
        }
        mortonCodes[i] = morton3D( xyz[0], xyz[1], xyz[2] );
    }
}

void calculateBoundingBoxOfTheScene( AABB const *boundingBoxes, int n,
                                     AABB &sceneBoundingBox )
{
    // QUESTION: precondition on sceneBoundingBox?
    for ( int i = 0; i < n; ++i ) // parallel reduce
        expand( sceneBoundingBox, boundingBoxes[i] );
}

// TODO: current implementation assumes all Morton codes are unique
// Need to handle duplicates expilcitly
Node *generateHierarchy( unsigned int *sortedMortonCodes, int *sortedObjectIDs,
                         int numObjects, LeafNode *leafNodes,
                         InternalNode *internalNodes )
{
    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.

    for ( int idx = 0; idx < numObjects; idx++ ) // in parallel
        leafNodes[idx].objectID = sortedObjectIDs[idx];

    // Construct internal nodes.

    for ( int idx = 0; idx < numObjects - 1; idx++ ) // in parallel
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        auto range = determineRange( sortedMortonCodes, numObjects, idx );
        int first = range.first;
        int last = range.second;
        //        std::cout<<idx<<"  range=("<<first<<", "<<last<<")\n";

        // Determine where to split the range.

        int split = findSplit( sortedMortonCodes, first, last );

        // Select childA.

        Node *childA;
        if ( split == first )
            childA = &leafNodes[split];
        else
            childA = &internalNodes[split];

        // Select childB.

        Node *childB;
        if ( split + 1 == last )
            childB = &leafNodes[split + 1];
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.

        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
    }

    // Node 0 is the root.

    return &internalNodes[0];
}

void calculateBoundingBoxes( LeafNode *leafNodes, InternalNode *internalNodes,
                             int numObjects, AABB *boundingBoxes )
{
    // possibly use Kokkos::atomic_fetch_add() here
    std::vector<std::atomic_flag> atomic_flags( numObjects - 1 );
    // flags are in an unspecified state on construction
    // their value cannot be copied/moved (constructor and assigment deleted)
    // so we have to loop over them and initialize them to the clear state
    for ( auto &flag : atomic_flags )
        flag.clear();

    auto getAABB = [boundingBoxes, numObjects, leafNodes,
                    internalNodes]( Node *node ) {
        int idx = -1;
        if ( auto leaf = dynamic_cast<LeafNode *>( node ) )
        {
            idx = leaf->objectID;
        }
        else
        {
            auto internal = dynamic_cast<InternalNode *>( node );
            idx = internal - internalNodes;
            idx += numObjects;
        }
        return boundingBoxes + idx;
    };

    InternalNode *root = internalNodes;
    for ( int i = 0; i < numObjects; ++i ) // parallel for
    {
        InternalNode *node =
            dynamic_cast<InternalNode *>( leafNodes[i].parent );
        do
        {
            if ( !atomic_flags[node - root].test_and_set() )
                break;
            expand( *getAABB( node ), *getAABB( node->childA ) );
            expand( *getAABB( node ), *getAABB( node->childB ) );
            node = dynamic_cast<InternalNode *>( node->parent );
        } while ( node != nullptr );
    }
}

} // end namespace Details
} // end namespace DataTransferKit
