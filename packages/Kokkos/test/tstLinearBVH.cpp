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
    Kokkos::View<DataTransferKit::AABB *, DeviceType, Kokkos::MemoryUnmanaged>
        bounding_boxes( bounding_boxes_vector.data(), n );
    DataTransferKit::BVH<NO> bvh( bounding_boxes );

    // (i) use same objects for the queries than the objects we constructed the
    // BVH
    for ( int i = 0; i < n; ++i )
    {
        Overlap overlap_predicate( bounding_boxes[i] );
        auto collision = details::spatial_query( bvh, overlap_predicate );
        // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
        TEST_EQUALITY( collision.size(), 1 );
        TEST_EQUALITY( *collision.begin(), i );
    }

    // (ii) use bounding boxes that overlap with first neighbors
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                // bounding box around nodes of the structured grid will overlap
                // with neighboring nodes
                bounding_boxes[ind( i, j, k )] = {
                    ( i - 1 ) * Lx / ( nx - 1 ), ( i + 1 ) * Lx / ( nx - 1 ),
                    ( j - 1 ) * Ly / ( ny - 1 ), ( j + 1 ) * Ly / ( ny - 1 ),
                    ( k - 1 ) * Lz / ( nz - 1 ), ( k + 1 ) * Lz / ( nz - 1 ),
                };
                // fill in reference solution to check againt the collision list
                // computed during the tree traversal
                std::set<int> ref;
                if ( ( i > 0 ) && ( j > 0 ) && ( k > 0 ) )
                    ref.emplace( ind( i - 1, j - 1, k - 1 ) );
                if ( ( i > 0 ) && ( k > 0 ) )
                    ref.emplace( ind( i - 1, j, k - 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref.emplace( ind( i - 1, j + 1, k - 1 ) );
                if ( ( i > 0 ) && ( j > 0 ) )
                    ref.emplace( ind( i - 1, j - 1, k ) );
                if ( i > 0 )
                    ref.emplace( ind( i - 1, j, k ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) )
                    ref.emplace( ind( i - 1, j + 1, k ) );
                if ( ( i > 0 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i - 1, j - 1, k + 1 ) );
                if ( ( i > 0 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i - 1, j, k + 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i - 1, j + 1, k + 1 ) );

                if ( ( j > 0 ) && ( k > 0 ) )
                    ref.emplace( ind( i, j - 1, k - 1 ) );
                if ( k > 0 )
                    ref.emplace( ind( i, j, k - 1 ) );
                if ( ( j < ny - 1 ) && ( k > 0 ) )
                    ref.emplace( ind( i, j + 1, k - 1 ) );
                if ( j > 0 )
                    ref.emplace( ind( i, j - 1, k ) );
                if ( true )
                    ref.emplace( ind( i, j, k ) );
                if ( j < ny - 1 )
                    ref.emplace( ind( i, j + 1, k ) );
                if ( ( j > 0 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i, j - 1, k + 1 ) );
                if ( k < nz - 1 )
                    ref.emplace( ind( i, j, k + 1 ) );
                if ( ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i, j + 1, k + 1 ) );

                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k > 0 ) )
                    ref.emplace( ind( i + 1, j - 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( k > 0 ) )
                    ref.emplace( ind( i + 1, j, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref.emplace( ind( i + 1, j + 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) )
                    ref.emplace( ind( i + 1, j - 1, k ) );
                if ( i < nx - 1 )
                    ref.emplace( ind( i + 1, j, k ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) )
                    ref.emplace( ind( i + 1, j + 1, k ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i + 1, j - 1, k + 1 ) );
                if ( ( i < nx - 1 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i + 1, j, k + 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref.emplace( ind( i + 1, j + 1, k + 1 ) );

                // traverse the tree and find potential collisions
                Overlap overlap_predicate( bounding_boxes[ind( i, j, k )] );
                auto collision =
                    details::spatial_query( bvh, overlap_predicate );
                // check the answer is the same as the reference we computed
                TEST_EQUALITY( collision.size(), ref.size() );
                for ( auto const &x : collision )
                    TEST_EQUALITY( ref.count( x ), 1 );
            }
    // (iii) use random points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );

    int nn = 1000;
    int count = 0; // drop point if mapped into [0.5-eps], 0.5+eps]^3
    for ( int l = 0; l < nn; ++l )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        DataTransferKit::AABB aabb;
        aabb = {
            x - 0.5 * Lx / ( nx - 1 ), x + 0.5 * Lx / ( nx - 1 ),
            y - 0.5 * Ly / ( ny - 1 ), y + 0.5 * Ly / ( ny - 1 ),
            z - 0.5 * Lz / ( nz - 1 ), z + 0.5 * Lz / ( nz - 1 ),
        };

        Overlap overlap_predicate( aabb );
        auto collision = details::spatial_query( bvh, overlap_predicate );
        int i = std::round( x / Lx * ( nx - 1 ) );
        int j = std::round( y / Ly * ( ny - 1 ) );
        int k = std::round( z / Lz * ( nz - 1 ) );
        // drop point if it the bounding box is going to overlap with more than
        // one
        // bounding box
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
        TEST_EQUALITY( collision.size(), 1 );
        TEST_EQUALITY( *collision.begin(), ind( i, j, k ) );
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
    using Point = bg::model::point<double, 3, bg::cs::cartesian>;
    using RTree = bgi::rtree<std::pair<Point, int>, bgi::linear<16>>;

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
        rtree.insert( std::make_pair( Point( x, y, z ), i ) );
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
    auto queries = make_random_cloud( Lx, Ly, Lz, 100 );
    // use random radius for the search and random number k of for the kNN
    // search
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_radius(
        0.0, std::sqrt( Lx * Lx + Ly * Ly + Lz * Lz ) );
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor( sqrt( nx * nx + ny * ny + nz * nz ) ) );
    for ( auto const &point : queries )
    {
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        Point centroid( x, y, z );
        double radius = distribution_radius( generator );
        int k = distribution_k( generator );

        // COMMENT: Did not implement proper radius search yet
        // This use available tree traversal for axis-aligned bounding box and
        // filters out candidates afterwards.
        // The coordinates of the points in the structured cloud (source) are
        // accessed directly and we use Boost to compute the distance.
        // This will need to be cleaned up, possibly with a templated tree
        // traversal with a predicate.
        details::Within within_predicate( {x, y, z}, radius );
        auto sol_within = details::spatial_query( bvh, within_predicate );

        // use the R-tree to obtain a reference solution
        std::vector<std::pair<Point, int>> returned_values;
        rtree.query( bgi::satisfies( [centroid, radius](
                         std::pair<Point, int> const &val ) {
                         return bg::distance( centroid, val.first ) <= radius;
                     } ),
                     std::back_inserter( returned_values ) );

        // I tried to encapsulate this in a lambda but it won't work with the
        // Teuchos test assertion macros
        // Anyway, this is fine for now...
        {
            auto const &ref = returned_values;
            TEST_EQUALITY( sol_within.size(), ref.size() );

            std::set<int> ref_ids;
            for ( auto const &id : ref )
                ref_ids.emplace( id.second );

            for ( auto const &id : sol_within )
                TEST_EQUALITY( ref_ids.count( id ), 1 );
        }

        // k nearest neighbors
        returned_values.clear();

        rtree.query( bgi::nearest( Point( x, y, z ), k ),
                     std::back_inserter( returned_values ) );
        std::list<std::pair<int, double>> sol;
        sol = DataTransferKit::Details::nearest_query( bvh, {x, y, z}, k );
        // copy/paste of the check solution against reference above
        {
            auto const &ref = returned_values;
            auto compare = []( std::pair<int, double> const &a,
                               std::pair<int, double> const &b ) {
                return a.first < b.first;
            };
            std::set<std::pair<int, double>, decltype( compare )> tmp(
                compare );
            tmp.insert( sol.begin(), sol.end() );
            TEST_EQUALITY( sol.size(), ref.size() );
            for ( auto const &x : ref )
            {
                auto it = tmp.find( std::make_pair( x.second, -1.0 ) );
                TEST_ASSERT( it != tmp.end() );
                TEST_EQUALITY( it->second, bg::distance( centroid, x.first ) );
            }
        }
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
