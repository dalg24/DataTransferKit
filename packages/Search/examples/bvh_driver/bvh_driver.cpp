/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "/scratch/source/trilinos/release/DataTransferKit/packages/Search/test/DTK_BoostRTreeHelpers.hpp"
#include <DTK_LinearBVH.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <random>

struct BoundingVolumeHierarchyTag
{
};

struct RTreeTag
{
};

struct KDTreeTag
{
};

template <typename Index>
struct Traits
{
    using Tag = BoundingVolumeHierarchyTag;
};

template <>
struct Traits<BoostRTreeHelpers::RTree<DataTransferKit::Box>>
{
    using Tag = RTreeTag;
};

template <typename Index, typename Indexable>
Index make_index_dispatch( Indexable geometries, BoundingVolumeHierarchyTag )
{
    return Index( geometries );
}

template <typename Index, typename Indexable>
Index make_index_dispatch( Indexable geometries, RTreeTag )
{
    return BoostRTreeHelpers::makeRTree( geometries );
}

template <typename Index, typename Indexable>
Index make_index( Indexable geometries )
{
    using Tag = typename Traits<Index>::Tag;
    return make_index_dispatch<Index, Indexable>( geometries, Tag{} );
}

template <typename Index, typename Query>
std::tuple<Kokkos::View<int *, typename Query::device_type>,
           Kokkos::View<int *, typename Query::device_type>>
query_dispatch( Index index, Query queries, BoundingVolumeHierarchyTag )
{
    Kokkos::View<int *, typename Query::device_type> offset( "offset" );
    Kokkos::View<int *, typename Query::device_type> indices( "indices" );
    index.query( queries, indices, offset );
    return std::make_tuple( offset, indices );
}

template <typename Index, typename Query>
std::tuple<Kokkos::View<int *, typename Query::device_type>,
           Kokkos::View<int *, typename Query::device_type>>
query_dispatch( Index index, Query queries, RTreeTag )
{
    return BoostRTreeHelpers::performQueries( index, queries );
}

template <typename Index, typename Query>
std::tuple<Kokkos::View<int *, typename Query::device_type>,
           Kokkos::View<int *, typename Query::device_type>>
query( Index index, Query queries )
{
    using Tag = typename Traits<Index>::Tag;
    return query_dispatch( index, queries, Tag{} );
}

template <class NO, typename Index>
int main_( Teuchos::CommandLineProcessor &clp, int argc, char *argv[] )
{
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    int n_values = 50000;
    int n_queries = 20000;
    int n_neighbors = 10;
    bool perform_knn_search = true;
    bool perform_radius_search = true;

    clp.setOption( "values", &n_values, "number of indexable values (source)" );
    clp.setOption( "queries", &n_queries, "number of queries (target)" );
    clp.setOption( "neighbors", &n_neighbors,
                   "desired number of results per query" );
    clp.setOption( "perform-knn-search", "do-not-perform-knn-search",
                   &perform_knn_search,
                   "whether or not to perform kNN search" );
    clp.setOption( "perform-radius-search", "do-not-perform-radius-search",
                   &perform_radius_search,
                   "whether or not to perform radius search" );

    clp.recogniseAllOptions( true );
    switch ( clp.parse( argc, argv ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        return EXIT_SUCCESS;
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        "random_points" );
    {
        // Random points are "reused" between building the tree and performing
        // queries.  You may change it if you have a problem with it.  These
        // don't really need to be stored in the 1st place.  What is needed is
        // indexable objects/values (here boxes) to build a tree and queries
        // (here kNN and radius searches) with mean to control the amount of
        // work per query as the problem size varies.
        auto n = std::max( n_values, n_queries );
        Kokkos::resize( random_points, n );

        auto random_points_host = Kokkos::create_mirror_view( random_points );

        // Generate random points uniformely distributed within a box.  The
        // edge length of the box chosen such that object density (here objects
        // will be boxes 2x2x2 centered around a random point) will remain
        // constant as problem size is changed.
        auto const a = std::cbrt( n_values );
        std::uniform_real_distribution<double> distribution( -a, +a );
        std::default_random_engine generator;
        auto random = [&distribution, &generator]() {
            return distribution( generator );
        };
        for ( int i = 0; i < n; ++i )
            random_points_host( i ) = {{random(), random(), random()}};
        Kokkos::deep_copy( random_points, random_points_host );
    }

    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        "bounding_boxes", n_values );
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                          KOKKOS_LAMBDA( int i ) {
                              double const x = random_points( i )[0];
                              double const y = random_points( i )[1];
                              double const z = random_points( i )[2];
                              bounding_boxes( i ) = {
                                  {{x - 1., y - 1., z - 1.}},
                                  {{x + 1., y + 1., z + 1.}}};
                          } );
    Kokkos::fence();

    std::ostream &os = std::cout;

    auto start = std::chrono::high_resolution_clock::now();
    auto index = make_index<Index>( bounding_boxes );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    os << "construction " << elapsed_seconds.count() << "\n";

    if ( perform_knn_search )
    {
        Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *,
                     DeviceType>
            queries( "queries", n_queries );
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                queries( i ) = DataTransferKit::nearest<DataTransferKit::Point>(
                    random_points( i ), n_neighbors );
            } );
        Kokkos::fence();

        start = std::chrono::high_resolution_clock::now();
        auto results = query( index, queries );
        end = std::chrono::high_resolution_clock::now();

        elapsed_seconds = end - start;
        os << "knn " << elapsed_seconds.count() << "\n";
    }

    if ( perform_radius_search )
    {
        Kokkos::View<DataTransferKit::Within *, DeviceType> queries(
            "queries", n_queries );
        // radius chosen in order to control the number of results per query
        // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
        // inserted into the tree (mid-point between half-edge and
        // half-diagonal)
        double const r = 2. * std::cbrt( static_cast<double>( n_neighbors ) *
                                         3. / ( 4. * M_PI ) ) -
                         ( 1. + std::sqrt( 3. ) ) / 2.;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                queries( i ) = DataTransferKit::within( random_points( i ), r );
            } );
        Kokkos::fence();

        start = std::chrono::high_resolution_clock::now();
        auto results = query( index, queries );
        end = std::chrono::high_resolution_clock::now();

        elapsed_seconds = end - start;
        os << "radius " << elapsed_seconds.count() << "\n";
    }

    return 0;
}

int main( int argc, char *argv[] )
{
    Kokkos::initialize( argc, argv );

    bool success = true;
    bool verbose = true;

    try
    {
        const bool throwExceptions = false;

        Teuchos::CommandLineProcessor clp( throwExceptions );

        std::string node = "";
        clp.setOption( "node", &node, "node type (serial | openmp | cuda)" );

        clp.recogniseAllOptions( false );
        switch ( clp.parse( argc, argv, NULL ) )
        {
        case Teuchos::CommandLineProcessor::PARSE_ERROR:
            success = false;
        case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
            break;
        }

        if ( !success )
        {
            // do nothing, just skip other if clauses
        }
        else if ( node == "" )
        {
            typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
            using DeviceType = typename Node::device_type;
            main_<Node, DataTransferKit::BoundingVolumeHierarchy<DeviceType>>(
                clp, argc, argv );
        }
        else if ( node == "serial" )
        {
#ifdef KOKKOS_ENABLE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
            using DeviceType = typename Node::device_type;
            main_<Node, DataTransferKit::BoundingVolumeHierarchy<DeviceType>>(
                clp, argc, argv );
#else
            throw std::runtime_error( "Serial node type is disabled" );
#endif
        }
        else if ( node == "openmp" )
        {
#ifdef KOKKOS_ENABLE_OPENMP
            typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
            using DeviceType = typename Node::device_type;
            main_<Node, DataTransferKit::BoundingVolumeHierarchy<DeviceType>>(
                clp, argc, argv );
#else
            throw std::runtime_error( "OpenMP node type is disabled" );
#endif
        }
        else if ( node == "cuda" )
        {
#ifdef KOKKOS_ENABLE_CUDA
            typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
            using DeviceType = typename Node::device_type;
            main_<Node, DataTransferKit::BoundingVolumeHierarchy<DeviceType>>(
                clp, argc, argv );
#else
            throw std::runtime_error( "CUDA node type is disabled" );
#endif
        }
        else if ( node == "boost" )
        {
#ifdef KOKKOS_ENABLE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
            using DeviceType = typename Node::device_type;
            main_<Node, BoostRTreeHelpers::RTree<DataTransferKit::Box>>(
                clp, argc, argv );
#else
            throw std::runtime_error( "CUDA node type is disabled" );
#endif
        }
        else
        {
            throw std::runtime_error( "Unrecognized node type" );
        }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS( verbose, std::cerr, success );

    Kokkos::finalize();

    return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
