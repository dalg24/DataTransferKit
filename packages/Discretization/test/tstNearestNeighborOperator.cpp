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
makeStructuredCloud( double Lx, double Ly, double Lz, int nx, int ny, int nz,
                     double offset_x = 0., double offset_y = 0.,
                     double offset_z = 0. )
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
                x = offset_x + i * Lx / ( nx - 1 );
                y = offset_y + j * Ly / ( ny - 1 );
                z = offset_z + k * Lz / ( nz - 1 );
                cloud[ind( i, j, k )] = {{x, y, z}};
            }
    return cloud;
}

std::vector<std::array<double, 3>>
makeRandomCloud( double Lx, double Ly, double Lz, int n, double seed = 0. )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator( seed );
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
void copyPointsFromCloud( std::vector<std::array<double, 3>> const &cloud,
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

    int const space_dim = 3;

    // Build structured cloud of points for the source and random cloud for the
    // target.
    Kokkos::View<double **, DeviceType> source_points( "source" );

    Kokkos::View<double **, DeviceType> target_points( "target", 1, space_dim );

    TEST_THROW( DataTransferKit::NearestNeighborOperator<DeviceType>(
                    comm, source_points, target_points ),
                DataTransferKit::DataTransferKitException );

    if ( comm_rank == 0 )
    {
        Kokkos::resize( source_points, 1, space_dim );
        auto source_points_host = Kokkos::create_mirror_view( source_points );
        for ( int d = 0; d < space_dim; ++d )
            source_points_host( 0, d ) = (double)comm_size;
        Kokkos::deep_copy( source_points, source_points_host );
    }

    auto target_points_host = Kokkos::create_mirror_view( target_points );
    for ( int d = 0; d < space_dim; ++d )
        target_points_host( 0, d ) = (double)comm_rank;
    Kokkos::deep_copy( target_points, target_points_host );

    // Shameless hack to help the distributed tree
    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::epsilon =
        (double)comm_size;

    DataTransferKit::NearestNeighborOperator<DeviceType> nnop(
        comm, source_points, target_points );

    Kokkos::View<double *, DeviceType> source_values( "in" );
    Kokkos::View<double *, DeviceType> target_values( "out" );

    // violate pre condition of apply
    TEST_THROW( nnop.apply( source_values, target_values ),
                DataTransferKit::DataTransferKitException );

    Kokkos::realloc( target_values, target_points.extent( 0 ) );
    Kokkos::realloc( source_values, source_points.extent( 0 ) );
    if ( comm_rank == 0 )
    {
        auto source_values_host = Kokkos::create_mirror_view( source_values );
        source_values_host( 0 ) = 255.;
        Kokkos::deep_copy( source_values, source_values_host );
    }

    nnop.apply( source_values, target_values );

    auto target_values_host = Kokkos::create_mirror_view( target_values );
    Kokkos::deep_copy( target_values_host, target_values );
    std::vector<double> target_values_ref = {255.};
    TEST_COMPARE_ARRAYS( target_values_host, target_values_ref );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( NearestNeighborOperator, hello_world, NODE )
{
    return;
    using DeviceType = typename NODE::device_type;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = comm->getRank();

    // Build structured cloud of points for the source and random cloud for the
    // target.
    Kokkos::View<double **, DeviceType> source_points( "source" );
    copyPointsFromCloud( makeStructuredCloud( 1.0, 1.0, 1.0, 11, 11, 11 ),
                         source_points );

    Kokkos::View<double **, DeviceType> target_points( "target" );
    copyPointsFromCloud( makeRandomCloud( 1.0, 1.0, 1.0, 20 ), target_points );

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

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( NearestNeighborOperator, structured_clouds,
                                   DeviceType )
{
    // The source is a structured cloud. The target is the same cloud but
    // distributed differently among the processors.
    Teuchos::RCP<Teuchos::Comm<int> const> comm =
        Teuchos::DefaultComm<int>::getComm();
    unsigned int const comm_size = comm->getSize();
    unsigned int const comm_rank = comm->getRank();

    // Build the structured cloud of points for the source and the target.
    double const L_x = 2.;
    double const L_y = 3.;
    double const L_z = 5.;
    unsigned int const n_x = 7;
    unsigned int const n_y = 11;
    unsigned int const n_z = 13;
    double const source_offset_x = comm_rank * L_x;
    double const source_offset_y = comm_rank * L_y;
    double const source_offset_z = comm_rank * L_z;

    Kokkos::View<double **, DeviceType> source_points( "source" );
    copyPointsFromCloud<DeviceType>(
        makeStructuredCloud( L_x, L_y, L_z, n_x, n_y, n_z, source_offset_x,
                             source_offset_y, source_offset_z ),
        source_points );

    double const target_offset_x = ( ( comm_rank + 1 ) % comm_size ) * L_x;
    double const target_offset_y = ( ( comm_rank + 1 ) % comm_size ) * L_y;
    double const target_offset_z = ( ( comm_rank + 1 ) % comm_size ) * L_z;

    Kokkos::View<double **, DeviceType> target_points( "target" );
    copyPointsFromCloud<DeviceType>(
        makeStructuredCloud( L_x, L_y, L_z, n_x, n_y, n_z, target_offset_x,
                             target_offset_y, target_offset_z ),
        target_points );

    unsigned int const n_points = source_points.extent( 0 );
    Kokkos::View<double *, DeviceType> target_values( "target_values",
                                                      n_points );

    DataTransferKit::NearestNeighborOperator<DeviceType> nnop(
        comm, source_points, target_points );

    // Wish I could use subview but Kokkos doesn't apply's interface
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<double *, DeviceType> source_values( "source_values",
                                                      n_points );
    Kokkos::parallel_for( "create_subview",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          KOKKOS_LAMBDA( int i ) {
                              source_values( i ) = source_points( i, 0 );
                          } );
    Kokkos::fence();

    nnop.apply( source_values, target_values );

    // Check results
    auto target_values_host = Kokkos::create_mirror_view( target_values );
    Kokkos::deep_copy( target_values_host, target_values );
    auto target_points_host = Kokkos::create_mirror_view( target_points );
    Kokkos::deep_copy( target_points_host, target_points );
    double const tol = 1e-14;
    for ( unsigned int i = 0; i < n_points; ++i )
        TEUCHOS_ASSERT( std::abs( target_values_host( i ) -
                                  target_points_host( i, 0 ) ) < tol );
}

void moveTargetPoints(
    std::vector<std::array<double, 3>> const &structured_cloud,
    double const L_x, double const L_y, double const L_z,
    unsigned int const n_x, unsigned int const n_y, unsigned int const n_z,
    double const offset_x, std::vector<std::array<double, 3>> &random_cloud,
    std::vector<unsigned int> &closest_points )
{
    closest_points.clear();
    double const half_x = L_x / ( 2 * n_x );
    double const half_y = L_y / ( 2 * n_y );
    double const half_z = L_z / ( 2 * n_z );

    for ( auto &point : random_cloud )
    {
        double x = point[0];
        double y = point[1];
        double z = point[2];

        // Move the point to the same domain covered by the structured cloud.
        bool done = false;
        while ( !done )
        {
            if ( x > L_x )
                x -= L_x;
            else
                done = true;
        }
        x += offset_x;

        // Find the closest point
        auto distance = [=]( std::array<double, 3> point ) {
            return std::sqrt( std::pow( point[0] - x, 2 ) +
                              std::pow( point[1] - y, 2 ) +
                              std::pow( point[2] - z, 2 ) );
        };

        double distance_min = std::numeric_limits<double>::max();
        unsigned int const n_points = structured_cloud.size();
        unsigned int source_point_index = 0;
        for ( unsigned int i = 0; i < n_points; ++i )
        {
            double d = distance( structured_cloud[i] );
            if ( d < distance_min )
            {
                distance_min = d;
                source_point_index = i;
            }
        }

        // Move the target point closer to the source point if necessary.
        auto nearest_point = structured_cloud[source_point_index];
        if ( std::abs( point[0] - nearest_point[0] ) >= half_x )
            point[0] = nearest_point[0] + half_x / 2.;
        if ( std::abs( point[1] - nearest_point[1] ) >= half_y )
            point[1] = nearest_point[1] + half_y / 2.;
        if ( std::abs( point[2] - nearest_point[2] ) >= half_z )
            point[2] = nearest_point[2] + half_z / 2.;

        closest_points.push_back( source_point_index );
    }
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( NearestNeighborOperator, mixed_clouds,
                                   DeviceType )
{
    // The source is a structured cloud. The target is a random cloud.
    Teuchos::RCP<Teuchos::Comm<int> const> comm =
        Teuchos::DefaultComm<int>::getComm();
    unsigned int const comm_size = comm->getSize();
    unsigned int const comm_rank = comm->getRank();

    // Build the structured cloud of points for the source.
    unsigned int constexpr spacedim = 3;
    double const L_x = 17.;
    double const L_y = 19.;
    double const L_z = 23.;
    unsigned int const n_x = 29;
    unsigned int const n_y = 31;
    unsigned int const n_z = 37;
    double const source_offset_x = comm_rank * L_x;
    double const source_offset_y = 0;
    double const source_offset_z = 0;

    std::vector<std::array<double, spacedim>> structured_cloud =
        makeStructuredCloud( L_x, L_y, L_z, n_x, n_y, n_z, source_offset_x,
                             source_offset_y, source_offset_z );
    Kokkos::View<double **, DeviceType> source_points( "source" );
    copyPointsFromCloud<DeviceType>( structured_cloud, source_points );

    // Build the random cloud of points for the target.
    unsigned int const n_target_points = 41;
    std::vector<std::array<double, spacedim>> random_cloud =
        makeRandomCloud( comm_size * L_x, comm_size * L_y, comm_size * L_z,
                         n_target_points, comm_rank );

    // Move the points to always be close to only one points of the source.
    std::vector<unsigned int> closest_points;
    moveTargetPoints( structured_cloud, L_x, L_y, L_z, n_x, n_y, n_z,
                      source_offset_x, random_cloud, closest_points );

    Kokkos::View<double **, DeviceType> target_points( "target" );
    copyPointsFromCloud<DeviceType>( random_cloud, target_points );

    Kokkos::View<double *, DeviceType> target_values( "target_values",
                                                      n_target_points );

    // Shameless hack to help the distributed tree
    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::epsilon =
        static_cast<double>( comm_size );

    DataTransferKit::NearestNeighborOperator<DeviceType> nnop(
        comm, source_points, target_points );

    using ExecutionSpace = typename DeviceType::execution_space;
    unsigned int const n_points = source_points.extent( 0 );
    Kokkos::View<double *, DeviceType> source_values( "source_values",
                                                      n_points );
    Kokkos::parallel_for( "create_subview",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          KOKKOS_LAMBDA( int i ) {
                              source_values( i ) = source_points( i, 0 );
                          } );
    Kokkos::fence();

    nnop.apply( source_values, target_values );

    // Check results
    auto target_values_host = Kokkos::create_mirror_view( target_values );
    Kokkos::deep_copy( target_values_host, target_values );
    double const tol = 1e-14;
    for ( unsigned int i = 0; i < n_target_points; ++i )
    {
        double ref_value = structured_cloud[closest_points[i]][0];

        TEUCHOS_ASSERT( std::abs( target_values_host( i, 0 ) - ref_value ) <
                        tol );
    }
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
                                          hello_world, NODE )                  \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(                                      \
        NearestNeighborOperator, structured_clouds, DeviceType##NODE )         \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( NearestNeighborOperator,             \
                                          mixed_clouds, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
