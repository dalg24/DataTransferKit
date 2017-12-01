/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_DetailsUtils.hpp> // exclusivePrefixSum, lastElement
#include <DTK_LinearBVH.hpp>
#include <DTK_Predicates.hpp>

#include <Kokkos_View.hpp>

#include <chrono>
#include <random>

#include <Teuchos_UnitTestHarness.hpp>

#include "DTK_BoostRTreeHelpers.hpp"
#include "DTK_NanoflannKDTreeHelpers.hpp"
#include "Search_UnitTestHelpers.hpp"

template <typename DeviceType>
struct DTKBVHHelpers
{
    using ExecutionSpace = typename DeviceType::execution_space;

    static DataTransferKit::BVH<DeviceType> makeLinearBVH(
        Kokkos::View<DataTransferKit::Point *, DeviceType> const &points )
    {
        auto const n_points = points.extent( 0 );
        Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes",
                                                                n_points );
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            KOKKOS_LAMBDA( int i ) {
                DataTransferKit::Details::expand( boxes( i ), points( i ) );
            } );
        Kokkos::fence();
        return DataTransferKit::BVH<DeviceType>( boxes );
    }

    template <typename Query>
    static std::tuple<Kokkos::View<int *, DeviceType>,
                      Kokkos::View<int *, DeviceType>>
    performQueries( DataTransferKit::BVH<DeviceType> const &bvh,
                    Kokkos::View<Query *, DeviceType> queries )
    {
        Kokkos::View<int *, DeviceType> indices( "indices" );
        Kokkos::View<int *, DeviceType> offset( "offset" );
        bvh.query( queries, indices, offset );
        return std::make_tuple( offset, indices );
    }
};

template <typename DeviceType>
Kokkos::View<DataTransferKit::Point *, DeviceType>
makeRandomCloud( double Lx, double Ly, double Lz, int n, double seed = 0. )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> points( "points", n );
    auto points_host = Kokkos::create_mirror_view( points );

    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distributionx( 0.0, Lz );
    std::uniform_real_distribution<double> distributiony( 0.0, Ly );
    std::uniform_real_distribution<double> distributionz( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double const x = distributionx( generator );
        double const y = distributiony( generator );
        double const z = distributionz( generator );
        points_host( i ) = {{x, y, z}};
    }

    Kokkos::deep_copy( points, points_host );
    return points;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
makeKNNQueries( double Lx, double Ly, double Lz, int n, double seed = 0. )
{
    Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
        queries( "queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );

    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distributionx( 0.0, Lz );
    std::uniform_real_distribution<double> distributiony( 0.0, Ly );
    std::uniform_real_distribution<double> distributionz( 0.0, Lz );
    std::uniform_int_distribution<int> distributionk(
        1, std::floor( std::sqrt( n ) ) );
    for ( int i = 0; i < n; ++i )
    {
        double const x = distributionx( generator );
        double const y = distributiony( generator );
        double const z = distributionz( generator );
        double const k = distributionk( generator );
        queries_host( i ) =
            DataTransferKit::nearest( DataTransferKit::Point{{x, y, z}}, k );
    }

    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Within *, DeviceType>
makeRadiusQueries( double Lx, double Ly, double Lz, int n, double seed = 0. )
{
    Kokkos::View<DataTransferKit::Within *, DeviceType> queries( "queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );

    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distributionx( 0., Lz );
    std::uniform_real_distribution<double> distributiony( 0., Ly );
    std::uniform_real_distribution<double> distributionz( 0., Lz );
    std::uniform_real_distribution<double> distributionr(
        0., std::cbrt( Lx * Ly * Lz / n * 3. / 4. / 3.14 ) );
    for ( int i = 0; i < n; ++i )
    {
        double const x = distributionx( generator );
        double const y = distributiony( generator );
        double const z = distributionz( generator );
        double const r = distributionr( generator );
        queries_host( i ) = DataTransferKit::within( {{x, y, z}}, r );
    }

    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
void validateResults(
    std::tuple<Kokkos::View<int *, DeviceType>,
               Kokkos::View<int *, DeviceType>> const &reference,
    std::tuple<Kokkos::View<int *, DeviceType>,
               Kokkos::View<int *, DeviceType>> const &other,
    bool &success, Teuchos::FancyOStream &out )
{
    TEST_COMPARE_ARRAYS( std::get<0>( reference ), std::get<0>( other ) );
    auto const offset = std::get<0>( reference );
    auto const n_queries = offset.extent_int( 0 ) - 1;
    auto extractAndSort = []( Kokkos::View<int *, DeviceType> const &v,
                              int begin, int end ) {
        std::vector<int> r( v.data() + begin, v.data() + end );
        std::sort( r.begin(), r.end() );
        return r;
    };
    for ( int i = 0; i < n_queries; ++i )
        TEST_COMPARE_ARRAYS( extractAndSort( std::get<1>( reference ),
                                             offset( i ), offset( i + 1 ) ),
                             extractAndSort( std::get<1>( other ), offset( i ),
                                             offset( i + 1 ) ) );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( PerformanceAssessment, aaa, DeviceType )
{
    int const n_points = 1000000;
    int const n_queries = 100000;

    auto start = std::chrono::system_clock::now();
    auto points = makeRandomCloud<DeviceType>( 1., 1., 1., n_points );
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "make " << n_points << " points " << elapsed_seconds.count()
              << "\n";

    // DTK
    start = std::chrono::system_clock::now();
    auto bvh = DTKBVHHelpers<DeviceType>::makeLinearBVH( points );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[SETUP] dtk " << elapsed_seconds.count() << "\n";

    // Boost
    start = std::chrono::system_clock::now();
    auto rtree = BoostRTreeHelpers::makeRTree( points );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[SETUP] boost " << elapsed_seconds.count() << "\n";

    // Nanoflann
    start = std::chrono::system_clock::now();
    auto kdtree = NanoflannKDTreeHelpers<DeviceType>::makeKDTree( points );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[SETUP] nanoflann " << elapsed_seconds.count() << "\n";

    // Make nearest neighbors queries
    start = std::chrono::system_clock::now();
    auto knn_queries = makeKNNQueries<DeviceType>( 1., 1., 1., n_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "make " << n_queries << " kNN queries "
              << elapsed_seconds.count() << "\n";

    start = std::chrono::system_clock::now();
    auto bvh_results =
        DTKBVHHelpers<DeviceType>::performQueries( bvh, knn_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[kNN SEARCH] dtk " << elapsed_seconds.count() << "\n";

    start = std::chrono::system_clock::now();
    auto rtree_results =
        BoostRTreeHelpers::performQueries( rtree, knn_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[kNN SEARCH] boost " << elapsed_seconds.count() << "\n";

    std::cout << "[kNN SEARCH] validate dtk against boost ... ";
    validateResults( bvh_results, rtree_results, success, out );
    std::cout << "OK\n";

    start = std::chrono::system_clock::now();
    auto kdtree_results = NanoflannKDTreeHelpers<DeviceType>::performQueries(
        kdtree, knn_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[kNN SEARCH] nanoflann " << elapsed_seconds.count() << "\n";

    std::cout << "[kNN SEARCH] validate dtk against nanoflann ... ";
    validateResults( bvh_results, kdtree_results, success, out );
    std::cout << "OK\n";

    // Make radius search queries
    start = std::chrono::system_clock::now();
    auto radius_queries =
        makeRadiusQueries<DeviceType>( 1., 1., 1., n_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "make " << n_queries << " radius queries "
              << elapsed_seconds.count() << "\n";

    start = std::chrono::system_clock::now();
    bvh_results =
        DTKBVHHelpers<DeviceType>::performQueries( bvh, radius_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[radius SEARCH] dtk " << elapsed_seconds.count() << "\n";

    start = std::chrono::system_clock::now();
    rtree_results = BoostRTreeHelpers::performQueries( rtree, radius_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[radius SEARCH] boost " << elapsed_seconds.count() << "\n";

    std::cout << "[radius SEARCH] validate dtk against boost ... ";
    validateResults( bvh_results, rtree_results, success, out );
    std::cout << "OK\n";

    start = std::chrono::system_clock::now();
    kdtree_results = NanoflannKDTreeHelpers<DeviceType>::performQueries(
        kdtree, radius_queries );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "[radius SEARCH] nanoflann " << elapsed_seconds.count()
              << "\n";

    std::cout << "[radius SEARCH] validate dtk against nanoflann ... ";
    validateResults( bvh_results, kdtree_results, success, out );
    std::cout << "OK\n";
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( PerformanceAssessment, aaa,          \
                                          DeviceType##NODE )                   \
// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
