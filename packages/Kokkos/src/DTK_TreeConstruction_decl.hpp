#ifndef DTK_TREECONSTRUCTION_DECL_HPP
#define DTK_TREECONSTRUCTION_DECL_HPP

#include "DTK_ConfigDefs.hpp"
#include <details/DTK_DetailsAlgorithms.hpp>

#include <DTK_Box.hpp>
#include <DTK_Node.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
namespace Details
{
/**
 * This structure contains all the functions used to build the BVH. All the
 * functions are static.
 */
template <typename NO>
struct TreeConstruction
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    static void calculateBoundingBoxOfTheScene(
        Kokkos::View<BBox const *, DeviceType> bounding_boxes,
        BBox &scene_bounding_box );

    // to assign the Morton code for a given object, we use the centroid point
    // of its bounding box, and express it relative to the bounding box of the
    // scene.
    static void
    assignMortonCodes( Kokkos::View<BBox const *, DeviceType> bounding_boxes,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       BBox const &scene_bounding_box );

    static void
    sortObjects( Kokkos::View<unsigned int *, DeviceType> morton_codes,
                 Kokkos::View<int *, DeviceType> object_ids );

    static Node *generateHierarchy(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes );

    static void
    calculateBoundingBoxes( Kokkos::View<Node *, DeviceType> leaf_nodes,
                            Kokkos::View<Node *, DeviceType> internal_nodes );

    KOKKOS_INLINE_FUNCTION
    static int commonPrefix( Kokkos::View<unsigned int *, DeviceType> k, int n,
                             int i, int j )
    {
        if ( j < 0 || j > n - 1 )
            return -1;
        // our construction algorithm relies on keys being unique so we handle
        // explicitly case of duplicate Morton codes by augmenting each key by a
        // bit
        // representation of its index.
        if ( k[i] == k[j] )
        {
            // countLeadingZeros( k[i] ^ k[j] ) == 32
            return 32 + Details::countLeadingZeros( i ^ j );
        }
        return Details::countLeadingZeros( k[i] ^ k[j] );
    }

    KOKKOS_FUNCTION
    static int
    findSplit( Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
               int first, int last );

    /**
     * Branchless sign function. Return 1 if @param x is greater than zero, 0 if
     * @param x is zero, and -1 if @param is less than zero.
     */
    // QUESTION: do we want to add it to DTK helpers?
    KOKKOS_INLINE_FUNCTION
    static int sgn( int x ) { return ( x > 0 ) - ( x < 0 ); }

    KOKKOS_FUNCTION
    static Kokkos::pair<int, int> determineRange(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int n,
        int i );
};
}
}

#endif
