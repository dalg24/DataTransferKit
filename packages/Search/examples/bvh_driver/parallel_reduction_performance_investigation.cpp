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

#include <DTK_DetailsTreeConstruction.hpp>
#include <DTK_LinearBVH.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_Random.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <random>

template <class NO>
int main_( Teuchos::CommandLineProcessor &clp, int argc, char *argv[] )
{
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    int n_values = 50000;
    bool build_bounding_volume_hierarchy = false;

    clp.setOption( "values", &n_values, "number of indexable values (source)" );
    clp.setOption( "build_bounding_volume_hierarchy",
                   "do-not-build_bounding_volume_hierarchy",
                   &build_bounding_volume_hierarchy,
                   "whether or not to build the bounding volume hierarchy" );

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
        Kokkos::resize( random_points, n_values );

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
        for ( int i = 0; i < n_values; ++i )
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

    {
        // parallel min reduce over unmanaged view
        Kokkos::View<double *, DeviceType> v( "v", 6 * n_values );
        Kokkos::Random_XorShift64_Pool<> pool( 12371 );
        Kokkos::fill_random( v, pool, 1.0 );
        auto start = std::chrono::high_resolution_clock::now();
        double result;
        Kokkos::Experimental::Min<double> reducer( result );
        Kokkos::parallel_reduce(
            "min_reduce",
            Kokkos::RangePolicy<ExecutionSpace>( 0, 6 * n_values ),
            KOKKOS_LAMBDA( int i, double &update ) {
                if ( v( i ) < update )
                    update = v( i );
            },
            reducer );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        os << "min reduce View double" << elapsed_seconds.count() << "\n";
    }

    {
        // parallel min reduce over unmanaged view
        auto start = std::chrono::high_resolution_clock::now();
        double result;
        Kokkos::View<double *, Kokkos::LayoutRight,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        v( reinterpret_cast<double *>( bounding_boxes.data() ), 6 * n_values );
        Kokkos::Experimental::Min<double> reducer( result );
        Kokkos::parallel_reduce(
            "min_reduce",
            Kokkos::RangePolicy<ExecutionSpace>( 0, 6 * n_values ),
            KOKKOS_LAMBDA( int i, double &update ) {
                if ( v( i ) < update )
                    update = v( i );
            },
            reducer );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        os << "min reduce reinterpreted unmanaged view "
           << elapsed_seconds.count() << "\n";
    }

    {
        // unmanaged view
        auto start = std::chrono::high_resolution_clock::now();
        Kokkos::View<double ***, Kokkos::LayoutRight,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        v( reinterpret_cast<double *>( bounding_boxes.data() ), n_values, 2,
           3 );
        DataTransferKit::Box b;
        for ( int d = 0; d < 3; ++d )
        {
            auto buffer_min = Kokkos::subview( v, Kokkos::ALL, 0, d );
            Kokkos::Experimental::Min<double> min_reducer( b.minCorner()[d] );
            Kokkos::parallel_reduce(
                "min_reduce_" + std::to_string( d ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                KOKKOS_LAMBDA( const int &i, double &min ) {
                    if ( buffer_min( i ) < min )
                        min = buffer_min( i );
                },
                min_reducer );

            auto buffer_max = Kokkos::subview( v, Kokkos::ALL, 1, d );
            Kokkos::Experimental::Max<double> max_reducer( b.maxCorner()[d] );
            Kokkos::parallel_reduce(
                "max_reduce_" + std::to_string( d ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                KOKKOS_LAMBDA( const int &i, double &max ) {
                    if ( buffer_max( i ) > max )
                        max = buffer_max( i );
                },
                max_reducer );
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        os << "unmanaged view " << elapsed_seconds.count() << "\n";
        std::cout << "( " << b.minCorner()[0] << ", " << b.minCorner()[1]
                  << ", " << b.minCorner()[2] << " )  ( " << b.maxCorner()[0]
                  << ", " << b.maxCorner()[1] << ", " << b.maxCorner()[2]
                  << " )\n";
    }

    {
        // unmanaged view with copies
        auto start = std::chrono::high_resolution_clock::now();
        Kokkos::View<double ***, Kokkos::LayoutRight,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        v( reinterpret_cast<double *>( bounding_boxes.data() ), n_values, 2,
           3 );
        DataTransferKit::Box b;
        Kokkos::View<double *, DeviceType> buffer( "buffer", n_values );
        for ( int d = 0; d < 3; ++d )
        {
            Kokkos::deep_copy( buffer,
                               Kokkos::subview( v, Kokkos::ALL, 0, d ) );
            Kokkos::Experimental::Min<double> min_reducer( b.minCorner()[d] );
            Kokkos::parallel_reduce(
                "min_reduce_" + std::to_string( d ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                KOKKOS_LAMBDA( const int &i, double &min ) {
                    if ( buffer( i ) < min )
                        min = buffer( i );
                },
                min_reducer );

            Kokkos::deep_copy( buffer,
                               Kokkos::subview( v, Kokkos::ALL, 1, d ) );
            Kokkos::Experimental::Max<double> max_reducer( b.maxCorner()[d] );
            Kokkos::parallel_reduce(
                "max_reduce_" + std::to_string( d ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                KOKKOS_LAMBDA( const int &i, double &max ) {
                    if ( buffer( i ) > max )
                        max = buffer( i );
                },
                max_reducer );
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        os << "unmanaged view with copies " << elapsed_seconds.count() << "\n";
        os << bounding_boxes[0].minCorner()[0] << "  " << v( 0, 0, 0 ) << "\n";
        os << bounding_boxes[0].minCorner()[1] << "  " << v( 0, 0, 1 ) << "\n";
        os << bounding_boxes[0].minCorner()[2] << "  " << v( 0, 0, 2 ) << "\n";
        os << bounding_boxes[0].maxCorner()[0] << "  " << v( 0, 1, 0 ) << "\n";
        os << bounding_boxes[0].maxCorner()[1] << "  " << v( 0, 1, 1 ) << "\n";
        os << bounding_boxes[0].maxCorner()[2] << "  " << v( 0, 1, 2 ) << "\n";
        os << bounding_boxes[1].minCorner()[0] << "  " << v( 1, 0, 0 ) << "\n";
        os << bounding_boxes[1].minCorner()[1] << "  " << v( 1, 0, 1 ) << "\n";
        os << bounding_boxes[1].minCorner()[2] << "  " << v( 1, 0, 2 ) << "\n";
        os << bounding_boxes[1].maxCorner()[0] << "  " << v( 1, 1, 0 ) << "\n";
        os << bounding_boxes[1].maxCorner()[1] << "  " << v( 1, 1, 1 ) << "\n";
        os << bounding_boxes[1].maxCorner()[2] << "  " << v( 1, 1, 2 ) << "\n";
        std::cout << "( " << b.minCorner()[0] << ", " << b.minCorner()[1]
                  << ", " << b.minCorner()[2] << " )  ( " << b.maxCorner()[0]
                  << ", " << b.maxCorner()[1] << ", " << b.maxCorner()[2]
                  << " )\n";
    }

    DataTransferKit::Box scene_bounding_box;
    auto start = std::chrono::high_resolution_clock::now();
    DataTransferKit::Details::TreeConstruction<
        DeviceType>::calculateBoundingBoxOfTheScene( bounding_boxes,
                                                     scene_bounding_box );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    os << "calculate scene bounding box " << elapsed_seconds.count() << "\n";

    auto b = scene_bounding_box;
    std::cout << "( " << b.minCorner()[0] << ", " << b.minCorner()[1] << ", "
              << b.minCorner()[2] << " )  ( " << b.maxCorner()[0] << ", "
              << b.maxCorner()[1] << ", " << b.maxCorner()[2] << " )\n";

    if ( build_bounding_volume_hierarchy )
    {
        auto start = std::chrono::high_resolution_clock::now();
        DataTransferKit::BVH<DeviceType> bvh( bounding_boxes );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        os << "construction " << elapsed_seconds.count() << "\n";
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
            main_<Node>( clp, argc, argv );
        }
        else if ( node == "serial" )
        {
#ifdef KOKKOS_HAVE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
            main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "Serial node type is disabled" );
#endif
        }
        else if ( node == "openmp" )
        {
#ifdef KOKKOS_HAVE_OPENMP
            typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
            main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "OpenMP node type is disabled" );
#endif
        }
        else if ( node == "cuda" )
        {
#ifdef KOKKOS_HAVE_CUDA
            typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
            main_<Node>( clp, argc, argv );
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
