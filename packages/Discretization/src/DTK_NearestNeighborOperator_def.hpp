/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_NEAREST_NEIGHBOR_OPERATOR_DEF_HPP
#define DTK_NEAREST_NEIGHBOR_OPERATOR_DEF_HPP

#include <DTK_DetailsNearestNeighborOperatorImpl.hpp>
#include <DTK_DistributedSearchTree.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
NearestNeighborOperator<DeviceType>::NearestNeighborOperator(
    Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
    Kokkos::View<Coordinate **, DeviceType> source_points,
    Kokkos::View<Coordinate **, DeviceType> target_points )
    : _comm( comm )
    , _indices( "indices" )
    , _ranks( "ranks" )
{
    setup( source_points, target_points );
}

template <typename DeviceType>
void NearestNeighborOperator<DeviceType>::setup(
    Kokkos::View<Coordinate **, DeviceType> source_points,
    Kokkos::View<Coordinate **, DeviceType> target_points )
{
    // NOTE: instead of checking the pre-condition that there is at least one
    // source point passed to one of the rank, we let the tree handle the
    // communication and just check that the tree is not empty.

    // Build distributed search tree over the source points.
    int const n_source_points = source_points.extent( 0 );
    Kokkos::View<Box *, DeviceType> boxes( "boxes", n_source_points );
    Kokkos::parallel_for(
        "make_boxes", Kokkos::RangePolicy<ExecutionSpace>( 0, n_source_points ),
        KOKKOS_LAMBDA( int i ) {
            Details::expand( boxes( i ),
                             {source_points( i, 0 ), source_points( i, 1 ),
                              source_points( i, 2 )} );
        } );
    Kokkos::fence();
    DistributedSearchTree<DeviceType> search_tree( _comm, boxes );

    // Tree must have at least one leaf, otherwise it makes little sense to the
    // search for nearest neighbors.
    DTK_CHECK( !search_tree.empty() );

    // Query nearest neighbor for all target points.
    int const n_target_points = target_points.extent( 0 );
    Kokkos::View<Details::Nearest *, DeviceType> nearest_queries(
        "nearest", n_target_points );
    Kokkos::parallel_for(
        "setup_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
        KOKKOS_LAMBDA( int i ) {
            nearest_queries( i ) =
                Details::nearest( {target_points( i, 0 ), target_points( i, 1 ),
                                   target_points( i, 2 )} );
        } );
    Kokkos::fence();

    // Perform the actual search.
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    search_tree.query( nearest_queries, indices, offset, ranks );

    // Check post-condition that we did find a nearest neighbor to all target
    // points.
    DTK_ENSURE( lastElement( offset ) == n_target_points );

    // Save results.
    // NOTE: we don't bother keeping `offset` around since it is just `[0, 1, 2,
    // ..., n_target_poins]`
    _indices = indices;
    _ranks = ranks;
}

template <typename DeviceType>
void NearestNeighborOperator<DeviceType>::apply(
    Kokkos::View<double *, DeviceType> source_values,
    Kokkos::View<double *, DeviceType> target_values )
{
    int const n_source_values = source_values.extent( 0 );
    int const n_target_values = target_values.extent( 0 );
    // Precondition that the target is properly sized
    DTK_REQUIRE( _indices.extent_int( 0 ) == n_target_values );
    // TODO: check the source as well

    Kokkos::View<int *, DeviceType> buffer_ranks =
        Kokkos::create_mirror( DeviceType(), _ranks );
    Kokkos::deep_copy( buffer_ranks, _ranks );
    Kokkos::View<int *, DeviceType> buffer_indices =
        Kokkos::create_mirror( DeviceType(), _indices );
    Kokkos::deep_copy( buffer_indices, _indices );

    Kokkos::View<double *, DeviceType> buffer_values( "values" );

    NearestNeighborOperatorImpl<DeviceType>::pullSourceValues(
        _comm, source_values, buffer_indices, buffer_ranks, buffer_values );

    NearestNeighborOperatorImpl<DeviceType>::pushTargetValues(
        _comm, target_values, buffer_indices, buffer_ranks, buffer_values );
}

} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_NEARESTNEIGHBOROPERATOR_INSTANT( NODE )                            \
    template class NearestNeighborOperator<typename NODE::device_type>;

#endif
