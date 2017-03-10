#ifndef DTK_DETAILSTREECONSTRUCTION_DECL_HPP
#define DTK_DETAILSTREECONSTRUCTION_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_AABB.hpp>
#include <DTK_Node.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
template <typename SC, typename LO, typename GO, typename NO>
struct TreeConstruction
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    // COMMENT: most of these could/should be protected function in BVH to avoid
    // passing all this data around
    static void calculateBoundingBoxOfTheScene( AABB const *bounding_boxes, int n,
                                         AABB &scene_bounding_box ) const;

    // to assign the Morton code for a given object, we use the centroid point
    // of
    // its bounding box, and express it relative to the bounding box of the
    // scene.
    static void
    assignMortonCodes( AABB const *bounding_boxes,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       int n, AABB const &scene_bounding_box ) const;

    static void sortObjects( Kokkos::View<unsigned int *, DeviceType> morton_codes,
                      Kokkos::View<int *, DeviceType> object_ids, int n ) const;

    static Node *generateHierarchy(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int n,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes ) const;

    static void
    calculateBoundingBoxes( Kokkos::View<Node *, DeviceType> leaf_nodes,
                            Kokkos::View<Node *, DeviceType> internal_nodes,
                            int n ) const;

    static int commonPrefix( Kokkos::View<unsigned int *, DeviceType> k, int n,
                             int i, int j );

    static int
    findSplit( Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
               int first, int last );

    // branchless sign function
    // QUESTION: do we want to add it to DTK helpers?
    static int sgn( int x ) { return ( x > 0 ) - ( x < 0 ); }

    static Kokkos::pair<int, int> determineRange(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int n,
        int i );
};
}

#endif
