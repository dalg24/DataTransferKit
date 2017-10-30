/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DETAILS_NEAREST_NEIGHBOR_OPERATOR_IMPL_HPP
#define DTK_DETAILS_NEAREST_NEIGHBOR_OPERATOR_IMPL_HPP

#include <DTK_DetailsDistributedSearchTreeImpl.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
struct NearestNeighborOperatorImpl
{
    using ExecutionSpace = typename DeviceType::execution_space;

    static void
    pullSourceValues( Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
                      Kokkos::View<double *, DeviceType> source_values,
                      Kokkos::View<int *, DeviceType> &buffer_indices,
                      Kokkos::View<int *, DeviceType> &buffer_ranks,
                      Kokkos::View<double *, DeviceType> &buffer_values )
    {
        int const n_exports = buffer_indices.extent( 0 );
        Tpetra::Distributor distributor( comm );
        int const n_imports =
            distributor.createFromSends( Teuchos::ArrayView<int>(
                buffer_ranks.data(), buffer_ranks.extent( 0 ) ) );

        Kokkos::View<int *, DeviceType> export_target_indices( "target_indices",
                                                               n_exports );
        Kokkos::parallel_for(
            "iota", Kokkos::RangePolicy<ExecutionSpace>( 0, n_exports ),
            KOKKOS_LAMBDA( int i ) { export_target_indices( i ) = i; } );
        Kokkos::fence();
        Kokkos::View<int *, DeviceType> import_target_indices( "target_indices",
                                                               n_imports );
        DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
            distributor, export_target_indices, import_target_indices );

        Kokkos::View<int *, DeviceType> export_source_indices = buffer_indices;
        Kokkos::View<int *, DeviceType> import_source_indices( "source_indices",
                                                               n_imports );
        DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
            distributor, export_source_indices, import_source_indices );

        Kokkos::View<int *, DeviceType> export_ranks( "ranks", n_exports );
        Kokkos::View<int *, DeviceType> import_ranks( "ranks", n_imports );
        int const comm_rank = comm->getRank();
        Kokkos::deep_copy( export_ranks, comm_rank );
        DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
            distributor, export_ranks, import_ranks );

        buffer_indices = import_target_indices;
        buffer_ranks = import_ranks;
        Kokkos::realloc( buffer_values, n_imports );
        Kokkos::parallel_for(
            "get_source_values",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_imports ),
            KOKKOS_LAMBDA( int i ) {
                buffer_values( i ) =
                    source_values( import_source_indices( i ) );
            } );
        Kokkos::fence();
    }

    static void
    pushTargetValues( Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
                      Kokkos::View<double *, DeviceType> target_values,
                      Kokkos::View<int *, DeviceType> &buffer_indices,
                      Kokkos::View<int *, DeviceType> &buffer_ranks,
                      Kokkos::View<double *, DeviceType> &buffer_values )
    {
        Tpetra::Distributor distributor( comm );
        int const n_imports =
            distributor.createFromSends( Teuchos::ArrayView<int>(
                buffer_ranks.data(), buffer_ranks.extent( 0 ) ) );

        Kokkos::View<double *, DeviceType> export_source_values = buffer_values;
        Kokkos::View<double *, DeviceType> import_source_values(
            "source_values", n_imports );
        DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
            distributor, export_source_values, import_source_values );

        Kokkos::View<int *, DeviceType> export_target_indices = buffer_indices;
        Kokkos::View<int *, DeviceType> import_target_indices( "target_indices",
                                                               n_imports );
        DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
            distributor, export_target_indices, import_target_indices );

        Kokkos::parallel_for(
            "set_target_values",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_imports ),
            KOKKOS_LAMBDA( int i ) {
                target_values( import_target_indices( i ) ) =
                    import_source_values( i );
            } );
        Kokkos::fence();
    }
};

} // end namespace DataTransferKit

#endif
