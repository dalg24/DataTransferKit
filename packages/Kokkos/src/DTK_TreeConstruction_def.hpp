#ifndef DTK_DETAILSTREECONSTRUCTION_DEF_HPP
#define DTK_DETAILSTREECONSTRUCTION_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <Kokkos_Sort.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>

namespace DataTransferKit
{
namespace Functor
{
using Box = AABB;
template <typename DeviceType>
class AssignMortonCodes
{
  public:
    AssignMortonCodes( Box const *bounding_boxes,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       Box const &scene_bounding_box )
        : _bounding_boxes( bounding_boxes )
        , _morton_codes( morton_codes )
        , _scene_bounding_box( scene_bounding_box )
    {
    }

    void operator()( int const i ) const
    {
        Details::Point xyz;
        double a, b;
        Details::centroid( _bounding_boxes[i], xyz );
        // scale coordinates with respect to bounding box of the scene
        for ( int d = 0; d < 3; ++d )
        {
            a = _scene_bounding_box[2 * d];
            b = _scene_bounding_box[2 * d + 1];
            xyz[d] = ( xyz[d] - a ) / ( b - a );
        }
        _morton_codes[i] = Details::morton3D( xyz[0], xyz[1], xyz[2] );
    }

  private:
    Box const *_bounding_boxes;
    Kokkos::View<unsigned int *, DeviceType> _morton_codes;
    Box const &_scene_bounding_box;
};

template <typename SC, typename LO, typename GO, typename NO>
class GenerateHierarchy
{
  public:
    using DeviceType = typename NO::device_type;
    GenerateHierarchy(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes, int n )
        : _sorted_morton_codes( sorted_morton_codes )
        , _leaf_nodes( leaf_nodes )
        , _internal_nodes( internal_nodes )
        , _n( n )
    {
    }

    // from "Thinking Parallel, Part III: Tree Construction on the GPU" by
    // Karras
    void operator()( int const i ) const
    {
        // Construct internal nodes.
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        auto range = TreeConstruction<SC, LO, GO, NO>::determineRange(
            _sorted_morton_codes, _n, i );
        int first = range.first;
        int last = range.second;

        // Determine where to split the range.

        int split = TreeConstruction<SC, LO, GO, NO>::findSplit(
            _sorted_morton_codes, first, last );

        // Select childA.

        Node *childA;
        if ( split == first )
            childA = &_leaf_nodes[split];
        else
            childA = &_internal_nodes[split];

        // Select childB.

        Node *childB;
        if ( split + 1 == last )
            childB = &_leaf_nodes[split + 1];
        else
            childB = &_internal_nodes[split + 1];

        // Record parent-child relationships.

        _internal_nodes[i].children.first = childA;
        _internal_nodes[i].children.second = childB;
        childA->parent = &_internal_nodes[i];
        childB->parent = &_internal_nodes[i];
    }

  private:
    Kokkos::View<unsigned int *, DeviceType> _sorted_morton_codes;
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
    int _n;
};

template <typename DeviceType>
class CalculateBoundingBoxes
{
  public:
    CalculateBoundingBoxes( Kokkos::View<Node *, DeviceType> leaf_nodes,
                            Node *root,
                            std::vector<std::atomic_flag> &atomic_flags )
        : _leaf_nodes( leaf_nodes )
        , _root( root )
        , _atomic_flags( atomic_flags )
    {
    }

    void operator()( int const i ) const
    {
        Node *node = _leaf_nodes[i].parent;
        while ( node != _root )
        {
            if ( !_atomic_flags[node - _root].test_and_set() )
                break;
            for ( Node *child : {node->children.first, node->children.second} )
                Details::expand( node->bounding_box, child->bounding_box );
            node = node->parent;
        }
        // NOTE: could stop at node != root and then just check that what we
        // computed earlier (bounding box of the scene) is indeed the union of
        // the two children.
    }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Node *_root;
    std::vector<std::atomic_flag> &_atomic_flags;
};
}

template <typename SC, typename LO, typename GO, typename NO>
void TreeConstruction<SC, LO, GO, NO>::calculateBoundingBoxOfTheScene(
    AABB const *bounding_boxes, int n, AABB &scene_bounding_box ) const
{
    Details::Functor::ExpandBoxWithBox functor( bounding_boxes );
    Kokkos::parallel_reduce( "calculate_bouding_of_the_scene",
                             Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                             functor, scene_bounding_box );
    Kokkos::fence();
}

template <typename SC, typename LO, typename GO, typename NO>
void TreeConstruction<SC, LO, GO, NO>::assignMortonCodes(
    AABB const *bounding_boxes,
    Kokkos::View<unsigned int *, DeviceType> morton_codes, int n,
    AABB const &scene_bounding_box ) const
{
    Functor::AssignMortonCodes<DeviceType> functor(
        bounding_boxes, morton_codes, scene_bounding_box );
    Kokkos::parallel_for( "assign_morton_codes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          functor );
    Kokkos::fence();
}

template <typename SC, typename LO, typename GO, typename NO>
void TreeConstruction<SC, LO, GO, NO>::sortObjects(
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Kokkos::View<int *, DeviceType> object_ids, int n ) const
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

template <typename SC, typename LO, typename GO, typename NO>
Node *TreeConstruction<SC, LO, GO, NO>::generateHierarchy(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int n,
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes ) const
{
    Functor::GenerateHierarchy<SC, LO, GO, NO> functor(
        sorted_morton_codes, leaf_nodes, internal_nodes, n );
    Kokkos::parallel_for( "generate_hierarchy",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n - 1 ),
                          functor );
    Kokkos::fence();

    // Node 0 is the root.
    return &( internal_nodes.data()[0] );
}

template <typename SC, typename LO, typename GO, typename NO>
void TreeConstruction<SC, LO, GO, NO>::calculateBoundingBoxes(
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes, int n ) const
{
    // possibly use Kokkos::atomic_fetch_add() here
    std::vector<std::atomic_flag> atomic_flags( n - 1 );
    // flags are in an unspecified state on construction
    // their value cannot be copied/moved (constructor and assigment deleted)
    // so we have to loop over them and initialize them to the clear state
    for ( auto &flag : atomic_flags )
        flag.clear();

    Node *root = &internal_nodes[0];

    Functor::CalculateBoundingBoxes<DeviceType> functor( leaf_nodes, root,
                                                         atomic_flags );
    Kokkos::parallel_for( "calculate_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          functor );
    Kokkos::fence();
}

template <typename SC, typename LO, typename GO, typename NO>
int TreeConstruction<SC, LO, GO, NO>::commonPrefix(
    Kokkos::View<unsigned int *, DeviceType> k, int n, int i, int j )
{
    if ( j < 0 || j > n - 1 )
        return -1;
    // our construction algorithm relies on keys being unique so we handle
    // explicitly case of duplicate Morton codes by augmenting each key by a bit
    // representation of its index.
    if ( k[i] == k[j] )
    {
        // countLeadingZeros( k[i] ^ k[j] ) == 32
        return 32 + Details::countLeadingZeros( i ^ j );
    }
    return Details::countLeadingZeros( k[i] ^ k[j] );
}

template <typename SC, typename LO, typename GO, typename NO>
int TreeConstruction<SC, LO, GO, NO>::findSplit(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int first,
    int last )
{
    // Identical Morton codes => split the range in the middle.

    unsigned int first_code = sorted_morton_codes[first];
    unsigned int last_code = sorted_morton_codes[last];

    if ( first_code == last_code )
        return ( first + last ) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int common_prefix = __clz( first_code ^ last_code );

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = ( step + 1 ) >> 1;     // exponential decrease
        int new_split = split + step; // proposed new position

        if ( new_split < last )
        {
            unsigned int split_code = sorted_morton_codes[new_split];
            int split_prefix = __clz( first_code ^ split_code );
            if ( split_prefix > common_prefix )
                split = new_split; // accept proposal
        }
    } while ( step > 1 );

    return split;
}

template <typename SC, typename LO, typename GO, typename NO>
Kokkos::pair<int, int> TreeConstruction<SC, LO, GO, NO>::determineRange(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int n, int i )
{
    using std::min;
    using std::max;
    // determine direction of the range (+1 or -1)
    int direction = sgn( commonPrefix( sorted_morton_codes, n, i, i + 1 ) -
                         commonPrefix( sorted_morton_codes, n, i, i - 1 ) );
    assert( direction == +1 || direction == -1 );
    // compute upper bound for the length of the range
    int max_step = 2;
    int common_prefix =
        commonPrefix( sorted_morton_codes, n, i, i - direction );
    // compute upper bound for the length of the range
    while ( commonPrefix( sorted_morton_codes, n, i,
                          i + direction * max_step ) > common_prefix )
    {
        max_step = max_step << 1;
    }
    // find the other end using binary search
    int split = 0;
    int step = max_step;
    do
    {
        step = step >> 1;
        if ( commonPrefix( sorted_morton_codes, n, i,
                           i + ( split + step ) * direction ) > common_prefix )
            split += step;
    } while ( step > 1 );
    int j = i + split * direction;
    return {min( i, j ), max( i, j )};
}
}

// Explicit instantiation macro
#define DTK_TREECONSTRUCTION_INSTANT( SCALAR, LO, GO, NODE )                   \
    template struct TreeConstruction<SCALAR, LO, GO, NODE>;

#endif
