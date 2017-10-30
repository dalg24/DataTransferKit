/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <Teuchos_UnitTestHarness.hpp>

#include <DTK_DetailsDistributedSearchTreeImpl.hpp>
#include <DTK_DistributedSearchTree.hpp>
#include <DTK_NearestNeighborOperator.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_Map.hpp>

#include <array>
#include <numeric>
#include <random>
#include <vector>

std::vector<std::array<double, 3>>
make_stuctured_cloud( double Lx, double Ly, double Lz, int nx, int ny, int nz )
{
    std::vector<std::array<double, 3>> cloud( nx * ny * nz );
    std::function<int( int, int, int )> ind = [nx, ny, nz]( int i, int j,
                                                            int k ) {
        return i + j * nx + k * ( nx * ny );
    };
    double x, y, z;
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                x = i * Lx / ( nx - 1 );
                y = j * Ly / ( ny - 1 );
                z = k * Lz / ( nz - 1 );
                cloud[ind( i, j, k )] = {{x, y, z}};
            }
    return cloud;
}

std::vector<std::array<double, 3>> make_random_cloud( double Lx, double Ly,
                                                      double Lz, int n )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        cloud[i] = {{x, y, z}};
    }
    return cloud;
}

template <typename DeviceType>
void copy_points_from_cloud( std::vector<std::array<double, 3>> const &cloud,
                             Kokkos::View<double **, DeviceType> &points )
{
    int const n_points = cloud.size();
    int const spatial_dim = 3;
    Kokkos::realloc( points, n_points, spatial_dim );
    auto points_host = Kokkos::create_mirror_view( points );
    for ( int i = 0; i < n_points; ++i )
        for ( int d = 0; d < spatial_dim; ++d )
            points_host( i, d ) = cloud[i][d];
    Kokkos::deep_copy( points, points_host );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( NearestNeighborOperator,
                                   find_me_a_better_name, NODE )
{
    using DeviceType = typename NODE::device_type;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_size = comm->getSize();
    int const comm_rank = comm->getRank();

    // Build structured cloud of points for the source and random cloud for the
    // target.
    Kokkos::View<double **, DeviceType> source_points( "source" );
    copy_points_from_cloud( make_stuctured_cloud( 1.0, 1.0, 1.0, 11, 11, 11 ),
                            source_points );

    Kokkos::View<double **, DeviceType> target_points( "target" );
    copy_points_from_cloud( make_random_cloud( 1.0, 1.0, 1.0, 20 ),
                            target_points );
    DataTransferKit::NearestNeighborOperator<DeviceType> nnop(
        comm, source_points, target_points );

    Kokkos::View<double *, DeviceType> source_values( "in" );
    Kokkos::View<double *, DeviceType> target_values( "out" );
    // violate pre condition of apply
    TEST_THROW( nnop.apply( source_values, target_values ),
                DataTransferKit::DataTransferKitException );

    Kokkos::realloc( target_values, target_points.extent( 0 ) );
    Kokkos::realloc( source_values, source_points.extent( 0 ) );
    nnop.apply( source_values, target_values );

    // violate post condition at the end of setup
    Kokkos::View<double **, DeviceType> no_source_points( "empty", 0 );
    TEST_THROW( DataTransferKit::NearestNeighborOperator<DeviceType>(
                    comm, no_source_points, target_points ),
                DataTransferKit::DataTransferKitException );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( NearestNeighborOperator, hello_world, NODE )
{
    using DeviceType = typename NODE::device_type;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_size = comm->getSize();
    int const comm_rank = comm->getRank();

    // Build structured cloud of points for the source and random cloud for the
    // target.
    Kokkos::View<double **, DeviceType> source_points( "source" );
    copy_points_from_cloud( make_stuctured_cloud( 1.0, 1.0, 1.0, 11, 11, 11 ),
                            source_points );

    Kokkos::View<double **, DeviceType> target_points( "target" );
    copy_points_from_cloud( make_random_cloud( 1.0, 1.0, 1.0, 20 ),
                            target_points );

    int const n_source_points = source_points.extent( 0 );
    int const n_target_points = target_points.extent( 0 );
    std::cout << "source " << n_source_points << "\n";
    std::cout << "target " << n_target_points << "\n";

    using ExecutionSpace = typename DeviceType::execution_space;

    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes",
                                                            n_source_points );
    Kokkos::parallel_for(
        "make_boxes", Kokkos::RangePolicy<ExecutionSpace>( 0, n_source_points ),
        KOKKOS_LAMBDA( int i ) {
            DataTransferKit::Details::expand(
                boxes( i ), {source_points( i, 0 ), source_points( i, 1 ),
                             source_points( i, 2 )} );
        } );
    Kokkos::fence();
    DataTransferKit::DistributedSearchTree<DeviceType> search_tree( comm,
                                                                    boxes );

    Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType>
        nearest_queries( "nearest", n_target_points );
    Kokkos::parallel_for(
        "setup_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
        KOKKOS_LAMBDA( int i ) {
            nearest_queries( i ) = DataTransferKit::Details::nearest(
                {target_points( i, 0 ), target_points( i, 1 ),
                 target_points( i, 2 )} );
        } );

    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    search_tree.query( nearest_queries, indices, offset, ranks );

    int const n_results = DataTransferKit::lastElement( offset );
    std::cout << "found " << n_results << "\n";
    using LocalOrdinal = int;
    using GlobalOrdinal = int;
    //    using Node = typename
    //    DataTransferKit::ParallelTraits<DeviceType>::TpetraNode;
    using Node = NODE;

    auto domain_map =
        Tpetra::createContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            n_source_points, comm );
    auto range_map =
        Tpetra::createContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            n_target_points, comm );

    using Scalar = double;

    TEUCHOS_ASSERT_EQUALITY( indices.extent_int( 0 ), n_target_points );
    Kokkos::View<int *, DeviceType> global_indices( "rows", n_target_points );
    auto local_map = range_map->getLocalMap();
    Kokkos::parallel_for(
        "fill_target_global_ids",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
        KOKKOS_LAMBDA( int i ) {
            global_indices( i ) = local_map.getGlobalElement( i );
        } );
    Kokkos::fence();

    // get global ids of source
    Tpetra::Distributor distributor( comm );
    int const n_imports = distributor.createFromSends(
        Teuchos::ArrayView<int>( ranks.data(), ranks.extent( 0 ) ) );
    Kokkos::View<int *, DeviceType> import_source_lids( "source_local_ids",
                                                        n_imports );
    Kokkos::View<int *, DeviceType> import_target_gids( "target_global_ids",
                                                        n_imports );
    Kokkos::View<int *, DeviceType> import_ranks( "ranks", n_imports );
    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
        distributor, indices, import_source_lids );
    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
        distributor, global_indices, import_target_gids );
    Kokkos::deep_copy( ranks, comm_rank );
    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
        distributor, ranks, import_ranks );

    Kokkos::View<int *, DeviceType> source_gids( "source_global_ids",
                                                 n_imports );
    local_map = domain_map->getLocalMap();
    Kokkos::parallel_for( "fill_source_global_ids",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_imports ),
                          KOKKOS_LAMBDA( int i ) {
                              source_gids( i ) = local_map.getGlobalElement(
                                  import_source_lids( i ) );
                          } );
    Kokkos::fence();

    auto kokkosViewToUniqueTeuchosArray =
        []( Kokkos::View<int *, DeviceType> ids ) {
            auto ids_host = Kokkos::create_mirror_view( ids );
            Kokkos::deep_copy( ids_host, ids );
            Teuchos::Array<int> out_array( Teuchos::ArrayView<int>(
                ids_host.data(), ids_host.extent( 0 ) ) );
            std::sort( out_array.begin(), out_array.end() );
            auto last = std::unique( out_array.begin(), out_array.end() );
            out_array.erase( last, out_array.end() );
            return out_array;
        };
    auto getIndexBase = []( Teuchos::Comm<int> const &comm,
                            Teuchos::Array<int> const &array ) {
        int index_base;
        Teuchos::reduceAll( comm, Teuchos::REDUCE_MIN,
                            !array.empty() ? array.front()
                                           : std::numeric_limits<int>::max(),
                            Teuchos::ptrFromRef( index_base ) );
        return index_base;
    };
    auto target_index_list =
        kokkosViewToUniqueTeuchosArray( import_target_gids );
    auto source_index_list = kokkosViewToUniqueTeuchosArray( source_gids );
    auto target_index_base = getIndexBase( *comm, target_index_list );
    auto source_index_base = getIndexBase( *comm, source_index_list );

    std::stringstream ss;
    ss << comm_rank << " ##  ";
    for ( auto const &x : target_index_list )
        ss << x << "  ";
    std::cout << ss.str() << "\n";

    auto row_map =
        Teuchos::rcp( new Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            target_index_list(), target_index_base, comm ) );

    auto column_map =
        Teuchos::rcp( new Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            source_index_list(), source_index_base, comm ) );

    // Matrix has exactly one entry per row.
    Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> matrix(
        row_map, column_map, 1 );
    for ( int i = 0; i < n_imports; ++i )
    {
        double val = 1.0;
        int row = import_target_gids( i );
        int col = source_gids( i );
        matrix.insertGlobalValues( row, 1, &val, &col );
    }
    matrix.fillComplete();
}

// Include the test macros.
#include "DataTransferKitDiscretization_ETIHelperMacros.h"

// Create the test group
/*
#define UNIT_TEST_GROUP( NODE )                                                \
using DeviceType##NODE = typename NODE::device_type;                       \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( NearestNeighborOperator,             \
                                          hello_world, DeviceType##NODE )
*/
#define UNIT_TEST_GROUP( NODE )                                                \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( NearestNeighborOperator,             \
                                          find_me_a_better_name, NODE )        \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( NearestNeighborOperator,             \
                                          hello_world, NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
