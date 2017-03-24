#include <details/DTK_DetailsTreeTraversal.hpp>
#include <details/DTK_Predicate.hpp>

#include <DTK_LinearBVH.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <random>

namespace details = DataTransferKit::Details;

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, tag_dispatching, NO )
{
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    int const n = 2;
    std::vector<DataTransferKit::AABB> boxes_vector = {{{0, 0, 0, 0, 0, 0}},
                                                       {{1, 1, 1, 1, 1, 1}}};
    Kokkos::View<DataTransferKit::AABB *, DeviceType, Kokkos::MemoryUnmanaged>
        boxes( boxes_vector.data(), n );
    DataTransferKit::BVH<NO> bvh( boxes );
    Kokkos::View<int *, DeviceType> results;
    bvh.query( details::nearest( details::Point{0, 0, 0}, 1 ), results );

    details::Within within_predicate( details::Point{0, 0, 0}, 0.5 );
    bvh.query( within_predicate, results );
}

class Overlap
{
  public:
    Overlap( DataTransferKit::AABB const &queryAABB )
        : _queryAABB( queryAABB )
    {
    }

    bool operator()( DataTransferKit::Node const *node ) const
    {
        return details::overlaps( node->bounding_box, _queryAABB );
    }

  private:
    DataTransferKit::AABB const &_queryAABB;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, structured_grid, NO )
{
    double Lx = 100.0;
    double Ly = 100.0;
    double Lz = 100.0;
    int constexpr nx = 11;
    int constexpr ny = 11;
    int constexpr nz = 11;
    int constexpr n = nx * ny * nz;
    std::function<int( int, int, int )> ind = [nx, ny, nz](
        int i, int j, int k ) { return i + j * nx + k * ( nx * ny ); };
    double eps = 1.0e-6;
    std::vector<DataTransferKit::AABB> bounding_boxes_vector( n );
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                bounding_boxes_vector[i + j * nx + k * ( nx * ny )] = {
                    i * Lx / ( nx - 1 ) - eps, i * Lx / ( nx - 1 ) + eps,
                    j * Ly / ( ny - 1 ) - eps, j * Ly / ( ny - 1 ) + eps,
                    k * Lz / ( nz - 1 ) - eps, k * Lz / ( nz - 1 ) + eps,
                };
            }

    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<DataTransferKit::AABB *, DeviceType, Kokkos::MemoryUnmanaged>
        bounding_boxes( bounding_boxes_vector.data(), n );
    DataTransferKit::BVH<NO> bvh( bounding_boxes );

    // (i) use same objects for the queries than the objects we constructed the
    // BVH
    Kokkos::View<int * [2], DeviceType> identity( "identity", n );

    auto check_identity = KOKKOS_LAMBDA( const int i )
    {
        Overlap overlap_predicate( bounding_boxes[i] );
        auto collision = details::spatial_query( bvh, overlap_predicate );
        identity( i, 0 ) = collision.size();
        identity( i, 1 ) = *collision.begin();
    };

    Kokkos::parallel_for( "check_identity",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          check_identity );

    // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( identity( i, 0 ), 1 );
        TEST_EQUALITY( identity( i, 1 ), i );
    }

    // (ii) use bounding boxes that overlap with first neighbors
    // Compute the reference solution.
    std::vector<std::set<int>> ref( n );
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int const index = ind( i, j, k );
                // bounding box around nodes of the structured grid will overlap
                // with neighboring nodes
                bounding_boxes[index] = {
                    ( i - 1 ) * Lx / ( nx - 1 ), ( i + 1 ) * Lx / ( nx - 1 ),
                    ( j - 1 ) * Ly / ( ny - 1 ), ( j + 1 ) * Ly / ( ny - 1 ),
                    ( k - 1 ) * Lz / ( nz - 1 ), ( k + 1 ) * Lz / ( nz - 1 ),
                };
                // fill in reference solution to check againt the collision list
                // computed during the tree traversal
                if ( ( i > 0 ) && ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k - 1 ) );
                if ( ( i > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j, k - 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k - 1 ) );
                if ( ( i > 0 ) && ( j > 0 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k ) );
                if ( i > 0 )
                    ref[index].emplace( ind( i - 1, j, k ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k ) );
                if ( ( i > 0 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k + 1 ) );
                if ( ( i > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j, k + 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k + 1 ) );

                if ( ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i, j - 1, k - 1 ) );
                if ( k > 0 )
                    ref[index].emplace( ind( i, j, k - 1 ) );
                if ( ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i, j + 1, k - 1 ) );
                if ( j > 0 )
                    ref[index].emplace( ind( i, j - 1, k ) );
                if ( true )
                    ref[index].emplace( ind( i, j, k ) );
                if ( j < ny - 1 )
                    ref[index].emplace( ind( i, j + 1, k ) );
                if ( ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i, j - 1, k + 1 ) );
                if ( k < nz - 1 )
                    ref[index].emplace( ind( i, j, k + 1 ) );
                if ( ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i, j + 1, k + 1 ) );

                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k ) );
                if ( i < nx - 1 )
                    ref[index].emplace( ind( i + 1, j, k ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k + 1 ) );
                if ( ( i < nx - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j, k + 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k + 1 ) );
            }

    Kokkos::View<int * [2], DeviceType> first_neighbor( "first_neighbor", n );

    auto check_first_neighbor = KOKKOS_LAMBDA( const int i )
    {
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int const index = ind( i, j, k );
                Overlap overlap_predicate( bounding_boxes[index] );
                auto collision =
                    details::spatial_query( bvh, overlap_predicate );
                first_neighbor( index, 0 ) = collision.size();
                // Only check the first element because we don't know how many
                // elements there are when we build the View. To check the other
                // points, we need to first compute all the points using Boost.
                // Then, we need to copy the points in a View and create another
                // view with the offset.
                first_neighbor( index, 1 ) = *collision.begin();
            }
    };

    Kokkos::parallel_for( "check_first_neighbor",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, nx ),
                          check_first_neighbor );

    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int index = ind( i, j, k );
                TEST_EQUALITY( first_neighbor( index, 0 ),
                               static_cast<int>( ref[index].size() ) );
                TEST_EQUALITY( ref[index].count( first_neighbor( index, 1 ) ),
                               1 );
            }

    // (iii) use random points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );

    int nn = 1000;
    int count = 0; // drop point if mapped into [0.5-eps], 0.5+eps]^3
    Kokkos::View<DataTransferKit::AABB *, ExecutionSpace> aabb( "aabb", nn );
    std::vector<int> indices( nn );
    for ( int l = 0; l < nn; ++l )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        aabb[l] = {
            x - 0.5 * Lx / ( nx - 1 ), x + 0.5 * Lx / ( nx - 1 ),
            y - 0.5 * Ly / ( ny - 1 ), y + 0.5 * Ly / ( ny - 1 ),
            z - 0.5 * Lz / ( nz - 1 ), z + 0.5 * Lz / ( nz - 1 ),
        };

        int i = std::round( x / Lx * ( nx - 1 ) );
        int j = std::round( y / Ly * ( ny - 1 ) );
        int k = std::round( z / Lz * ( nz - 1 ) );
        // drop point if it the bounding box is going to overlap with more than
        // one bounding box
        if ( ( std::abs( x / Lx * ( nx - 1 ) -
                         std::floor( x / Lx * ( nx - 1 ) ) - 0.5 ) < eps ) ||
             ( std::abs( y / Ly * ( ny - 1 ) -
                         std::floor( y / Ly * ( ny - 1 ) ) - 0.5 ) < eps ) ||
             ( std::abs( z / Lz * ( nz - 1 ) -
                         std::floor( z / Lz * ( nz - 1 ) ) - 0.5 ) < eps ) )
        {
            ++count;
            continue;
        }
        // Save the indices for the check
        indices[l] = ind( i, j, k );
    }

    Kokkos::View<int * [2], DeviceType> random( "random", n );
    auto check_random = KOKKOS_LAMBDA( const int i )
    {

        Overlap overlap_predicate( aabb[i] );
        auto collision = details::spatial_query( bvh, overlap_predicate );
        random( i, 0 ) = collision.size();
        random( i, 1 ) = *collision.begin();
    };

    Kokkos::parallel_for( "check_random",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, nn ),
                          check_random );

    for ( int i = 0; i < nn; ++i )
    {
        TEST_EQUALITY( random( i, 0 ), 1 );
        TEST_EQUALITY( random( i, 1 ), indices[i] );
    }

    // make sure we did not drop all points
    TEST_COMPARE( count, <, n );
}

std::vector<std::array<double, 3>>
make_stuctured_cloud( double Lx, double Ly, double Lz, int nx, int ny, int nz )
{
    std::vector<std::array<double, 3>> cloud( nx * ny * nz );
    std::function<int( int, int, int )> ind = [nx, ny, nz](
        int i, int j, int k ) { return i + j * nx + k * ( nx * ny ); };
    double x, y, z;
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                x = i * Lx / ( nx - 1 );
                y = j * Ly / ( ny - 1 );
                z = k * Lz / ( nz - 1 );
                cloud[ind( i, j, k )] = {x, y, z};
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
        cloud[i] = {x, y, z};
    }
    return cloud;
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, rtree, NO )
{
    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    using BPoint = bg::model::point<double, 3, bg::cs::cartesian>;
    using RTree = bgi::rtree<std::pair<BPoint, int>, bgi::linear<16>>;

    // contruct a cloud of points (nodes of a structured grid)
    double Lx = 10.0;
    double Ly = 10.0;
    double Lz = 10.0;
    int nx = 11;
    int ny = 11;
    int nz = 11;
    auto cloud = make_stuctured_cloud( Lx, Ly, Lz, nx, ny, nz );
    int n = cloud.size();

    // create a R-tree to compare radius search results against
    RTree rtree;
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        rtree.insert( std::make_pair( BPoint( x, y, z ), i ) );
    }

    // build bounding volume hierarchy
    std::vector<DataTransferKit::AABB> bounding_boxes_vector( n );
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        bounding_boxes_vector[i] = {
            x, x, y, y, z, z,
        };
    }
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    Kokkos::View<DataTransferKit::AABB *, DeviceType, Kokkos::MemoryUnmanaged>
        bounding_boxes( bounding_boxes_vector.data(), n );
    DataTransferKit::BVH<NO> bvh( bounding_boxes );

    // random points for radius search and kNN queries
    // compare our solution against Boost R-tree
    int const n_points = 100;
    auto queries = make_random_cloud( Lx, Ly, Lz, n_points );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<double * [3], ExecutionSpace> point_coords( "point_coords",
                                                             n_points );
    Kokkos::View<double *, ExecutionSpace> radii( "radii", n_points );
    Kokkos::View<int * [2], ExecutionSpace> within_n_pts( "within_n_pts",
                                                          n_points );
    Kokkos::View<int * [2], ExecutionSpace> nearest_n_pts( "nearest_n_pts",
                                                           n_points );
    Kokkos::View<int *, ExecutionSpace> k( "distribution_k", n_points );
    std::vector<std::vector<std::pair<BPoint, int>>> returned_values_within(
        n_points );
    std::vector<std::vector<std::pair<BPoint, int>>> returned_values_nearest(
        n_points );
    // use random radius for the search and random number k of for the kNN
    // search
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_radius(
        0.0, std::sqrt( Lx * Lx + Ly * Ly + Lz * Lz ) );
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor( sqrt( nx * nx + ny * ny + nz * nz ) ) );
    for ( unsigned int i = 0; i < n_points; ++i )
    {
        auto const &point = queries[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        BPoint centroid( x, y, z );
        radii[i] = distribution_radius( generator );
        k[i] = distribution_k( generator );
        double radius = radii[i];

        // COMMENT: Did not implement proper radius search yet
        // This use available tree traversal for axis-aligned bounding box and
        // filters out candidates afterwards.
        // The coordinates of the points in the structured cloud (source) are
        // accessed directly and we use Boost to compute the distance.
        point_coords( i, 0 ) = x;
        point_coords( i, 1 ) = y;
        point_coords( i, 2 ) = z;

        // use the R-tree to obtain a reference solution
        rtree.query( bgi::satisfies( [centroid, radius](
                         std::pair<BPoint, int> const &val ) {
                         return bg::distance( centroid, val.first ) <= radius;
                     } ),
                     std::back_inserter( returned_values_within[i] ) );

        // k nearest neighbors
        rtree.query( bgi::nearest( BPoint( x, y, z ), k[i] ),
                     std::back_inserter( returned_values_nearest[i] ) );
    }

    auto random_within_lambda = KOKKOS_LAMBDA( const int i )
    {
        details::Within within_predicate(
            {point_coords( i, 0 ), point_coords( i, 1 ), point_coords( i, 2 )},
            radii( i ) );
        auto sol_within = details::spatial_query( bvh, within_predicate );
        within_n_pts( i, 0 ) = sol_within.size();
        within_n_pts( i, 1 ) = *sol_within.begin();
    };

    Kokkos::parallel_for( "random_within",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          random_within_lambda );

    for ( int i = 0; i < n_points; ++i )
    {
        auto const &ref = returned_values_within[i];
        TEST_EQUALITY( within_n_pts( i, 0 ), static_cast<int>( ref.size() ) );
        std::set<int> ref_ids;
        for ( auto const &id : ref )
            ref_ids.emplace( id.second );

        if ( ref.size() > 0 )
            TEST_EQUALITY( ref_ids.count( within_n_pts( i, 1 ) ), 1 );
    }

    auto random_nearest_lambda = KOKKOS_LAMBDA( const int i )
    {
        auto sol_nearest = DataTransferKit::Details::nearest_query(
            bvh,
            {point_coords( i, 0 ), point_coords( i, 1 ), point_coords( i, 2 )},
            k[i] );
        nearest_n_pts( i, 0 ) = sol_nearest.size();
        nearest_n_pts( i, 1 ) = sol_nearest.begin()->first;
    };

    Kokkos::parallel_for( "random_nearest",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          random_nearest_lambda );

    for ( int i = 0; i < n_points; ++i )
    {
        auto const &ref = returned_values_nearest[i];
        TEST_EQUALITY( nearest_n_pts( i, 0 ), static_cast<int>( ref.size() ) );
        std::set<int> ref_ids;
        for ( auto const &id : ref )
            ref_ids.emplace( id.second );

        TEST_EQUALITY( ref_ids.count( nearest_n_pts( i, 1 ) ), 1 );
    }
}

// Include the test macros.
#include "DataTransferKitKokkos_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, tag_dispatching, NODE )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, structured_grid, NODE )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, rtree, NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
