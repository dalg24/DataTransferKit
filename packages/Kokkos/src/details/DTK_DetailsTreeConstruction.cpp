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

// TODO: this is a mess
// we need a default impl
#define __clz( x ) __builtin_clz( x )
// default implementation if nothing else is available
// Taken from:
// http://stackoverflow.com/questions/23856596/counting-leading-zeros-in-a-32-bit-unsigned-integer-with-best-algorithm-in-c-pro
// WARNING: this implementation does **not** support __clz(0) (result should be
// 32 but this function returns 0)
int clz( uint32_t x )
{
    static const char debruijn32[32] = {
        0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
        1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return debruijn32[x * 0x076be629 >> 27];
}

// TODO: use preprocessor directive to select an implementation
// it turns out NVDIA's implementation of int __clz(unsigned int x) is
// slightly different than GCC __builtin_clz
// this caused a bug in an early implementation of the function that compute
// the common prefixes between two keys (NB: when i == j)
int countLeadingZeros( unsigned int x )
{
#if defined __GNUC__
    // int __builtin_clz(unsigned int x) result is undefined if x is 0
    return x != 0 ? __builtin_clz( x ) : 32;
#else
    // similar problem with the default implementation
    return x != 0 ? clz( x ) : 32;
#endif
}

int commonPrefix( unsigned int const *k, int n, int i, int j )
{
    if ( j < 0 || j > n - 1 )
        return -1;
    // our construction algorithm relies on keys being unique so we handle
    // explicitly case of duplicate Morton codes by augmenting each key by a bit
    // representation of its index.
    if ( k[i] == k[j] )
    {
        // countLeadingZeros( k[i] ^ k[j] ) == 32
        return 32 + countLeadingZeros( i ^ j );
    }
    return countLeadingZeros( k[i] ^ k[j] );
}

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

// to assign the Morton code for a given object, we use the centroid point of
// its bounding box, and express it relative to the bounding box of the scene.
void assignMortonCodes( AABB const *boundingBoxes, unsigned int *mortonCodes,
                        int n, AABB const &sceneBoundingBox )
{
    std::array<double, 3> xyz;
    double a, b;
    for ( int i = 0; i < n; ++i ) // parallel for
    {
        centroid(boundingBoxes[i], xyz);
        // scale coordinates with respect to bounding box of the scene
        for ( int d = 0; d < 3; ++d )
        {
            a = sceneBoundingBox[2 * d + 0];
            b = sceneBoundingBox[2 * d + 1];
            xyz[d] = ( xyz[d] - a ) / ( b - a );
        }
        mortonCodes[i] = morton3D( xyz[0], xyz[1], xyz[2] );
    }
}

void sortObjects( unsigned int *morton_codes, int *object_ids, int n )
{
    using std::sort;
    // possibly use thrust::sort()
    // see https://thrust.github.io
    sort( object_ids, object_ids + n,
          [morton_codes]( int const &i, int const &j ) {
              return morton_codes[i] < morton_codes[j];
          } );
    // TODO: in-place permutation of mortonCodes rather than 2nd sort
    sort( morton_codes, morton_codes + n );
}

void calculateBoundingBoxOfTheScene( AABB const *boundingBoxes, int n,
                                     AABB &sceneBoundingBox )
{
    // QUESTION: precondition on sceneBoundingBox?
    for ( int i = 0; i < n; ++i ) // parallel reduce
        expand( sceneBoundingBox, boundingBoxes[i] );
}

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
        // NOTE: could stop at node != root and then just check that what we
        // computed earlier (bounding box of the scene) is indeed the union of
        // the two children.
    }
}

} // end namespace Details
} // end namespace DataTransferKit
