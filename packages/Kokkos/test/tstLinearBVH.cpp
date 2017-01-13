#include <details/DTK_DetailsTreeTraversal.hpp>

#include <DTK_LinearBVH.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <random>

TEUCHOS_UNIT_TEST( LinearBVH, random_spheres )
{
    // create a sequence of spheres with centroid within the cube [-10,10]^3 and
    // radius in [1,3]
    int const n = 10;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_centroid( -10.0, 10.0 );
    std::uniform_real_distribution<double> distribution_radius( 1, 3 );
    std::vector<DataTransferKit::AABB> boundingBoxes( n );
    for ( int i = 0; i < n; ++i )
    {
        double centroid[3] = {
            distribution_centroid( generator ),
            distribution_centroid( generator ),
            distribution_centroid( generator ),
        };
        double radius = distribution_radius( generator );
        for ( int d = 0; d < 3; ++d )
        {
            boundingBoxes[i]._minmax[2 * d + 0] = centroid[d] - radius;
            boundingBoxes[i]._minmax[2 * d + 1] = centroid[d] + radius;
        }
    }

    DataTransferKit::BVH bvh( boundingBoxes.data(), n );

    //    std::cout << "sorted ids and morton codes\n";
    //    for ( int i = 0; i < n; ++i )
    //        std::cout << bvh._ids[i] << "  " << bvh._morton_indices[i] << "  "
    //                  << std::bitset<32>( bvh._morton_indices[i] ) << "\n";
    std::cout << "root bounding box = ";
    for ( auto const &x : bvh._bounding_boxes[n]._minmax )
        std::cout << x << "  ";
    std::cout << "\n";
    std::cout << "scene bounding box = ";
    for ( auto const &x : bvh._scene_bounding_box._minmax )
        std::cout << x << "  ";
    std::cout << "\n";

    for ( int d = 0; d < 3; ++d )
    {
        TEST_EQUALITY( bvh._bounding_boxes[n]._minmax[2 * d + 0],
                       bvh._scene_bounding_box._minmax[2 * d + 0] );
        TEST_EQUALITY( bvh._bounding_boxes[n]._minmax[2 * d + 1],
                       bvh._scene_bounding_box._minmax[2 * d + 1] );
    }

    DataTransferKit::Details::CollisionList collision_list;
    DataTransferKit::Details::traverseIterative( collision_list, bvh,
                                                 bvh._bounding_boxes[0], 0 );
    DataTransferKit::Details::traverseRecursive(
        collision_list, bvh, bvh._bounding_boxes[0], 0, bvh.getRoot() );
}

TEUCHOS_UNIT_TEST( LinearBVH, structured_grid )
{
    double Lx = 100.0;
    double Ly = 100.0;
    double Lz = 100.0;
    int nx = 11;
    int ny = 11;
    int nz = 11;
    std::function<int( int, int, int )> ind = [nx, ny, nz](
        int i, int j, int k ) { return i + j * nx + k * ( nx * ny ); };
    double eps = 1.0e-6;
    std::vector<DataTransferKit::AABB> bounding_boxes( nx * ny * nz );
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                bounding_boxes[i + j * nx + k * ( nx * ny )]._minmax = {
                    i * Lx / ( nx - 1 ) - eps, i * Lx / ( nx - 1 ) + eps,
                    j * Ly / ( ny - 1 ) - eps, j * Ly / ( ny - 1 ) + eps,
                    k * Lz / ( nz - 1 ) - eps, k * Lz / ( nz - 1 ) + eps,
                };
            }

    DataTransferKit::Details::CollisionList collision_list;
    DataTransferKit::BVH bvh( bounding_boxes.data(), nx * ny * nz );

    // (i) use same objects for the queries than the objects we constructed the
    // BVH
    for ( int i = 0; i < nx * ny * nz; ++i )
        //    DataTransferKit::traverseRecursive(collision_list, bvh,
        //    bvh._bounding_boxes[i], i, bvh.getRoot());
        DataTransferKit::Details::traverseIterative( collision_list, bvh,
                                                     bounding_boxes[i], i );

    // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
    TEST_EQUALITY( static_cast<int>( collision_list._ij.size() ),
                   nx * ny * nz );
    for ( int i = 0; i < nx * ny * nz; ++i )
    {
        TEST_EQUALITY( collision_list._ij[i].first, i )
        TEST_EQUALITY( collision_list._ij[i].first,
                       collision_list._ij[i].second );
    }

    // (ii) use bounding boxes that overlap with first neighbors
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                // bounding box around nodes of the structured grid will overlap
                // with
                // neighboring nodes
                bounding_boxes[ind( i, j, k )]._minmax = {
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
                collision_list._ij.clear();
                DataTransferKit::Details::traverseIterative(
                    collision_list, bvh, bounding_boxes[ind( i, j, k )],
                    ind( i, j, k ) );
                // check the answer is the same as the reference we computed
                TEST_EQUALITY( collision_list._ij.size(), ref.size() );
                for ( auto const &x : collision_list._ij )
                {
                    TEST_EQUALITY( ref.count( x.second ), 1 );
                    TEST_EQUALITY( x.first, ind( i, j, k ) );
                }
            }
    // (iii) use random points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );

    int n = 1000;
    int count = 0; // drop point if mapped into [0.5-eps], 0.5+eps]^3
    for ( int l = 0; l < n; ++l )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        DataTransferKit::AABB aabb;
        aabb._minmax = {
            x - 0.5 * Lx / ( nx - 1 ), x + 0.5 * Lx / ( nx - 1 ),
            y - 0.5 * Ly / ( ny - 1 ), y + 0.5 * Ly / ( ny - 1 ),
            z - 0.5 * Lz / ( nz - 1 ), z + 0.5 * Lz / ( nz - 1 ),
        };
        collision_list._ij.clear();
        DataTransferKit::Details::traverseIterative( collision_list, bvh, aabb,
                                                     -1 );
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
        TEST_EQUALITY( collision_list._ij.size(), 1 );
        TEST_EQUALITY( collision_list._ij[0].first, -1 );
        TEST_EQUALITY( collision_list._ij[0].second, ind( i, j, k ) );
    }
    // make sure we did not drop all points
    TEST_COMPARE( count, <, n );

    for ( auto const &x : collision_list._ij )
        std::cout << " (" << x.first << ", " << x.second << ") ";
    std::cout << "\n";
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

TEUCHOS_UNIT_TEST( LinearBVH, rtree )
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
    std::vector<DataTransferKit::AABB> bounding_boxes( n );
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        bounding_boxes[i]._minmax = {
            x, x, y, y, z, z,
        };
    }
    DataTransferKit::BVH bvh( bounding_boxes.data(), n );

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
        // traversal
        // with a predicate.
        std::list<std::pair<int, double>> sol;
        DataTransferKit::AABB aabb;
        aabb._minmax = {
            x - radius, x + radius, y - radius,
            y + radius, z - radius, z + radius,
        };
        DataTransferKit::Details::CollisionList collision_list;
        DataTransferKit::Details::traverseIterative( collision_list, bvh, aabb,
                                                     -1 );
        // filter out candidates, don't bother sorting the results
        for ( auto const &collision : collision_list._ij )
        {
            int p = collision.second;
            // here using directly cloud to make our life easier
            double x = std::get<0>( cloud[p] );
            double y = std::get<1>( cloud[p] );
            double z = std::get<2>( cloud[p] );
            Point candidate( x, y, z );
            double d = bg::distance( centroid, candidate );
            if ( d <= radius )
                sol.emplace_back( std::make_pair( p, d ) );
        }

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

        // k nearest neighbors
        sol.clear();
        returned_values.clear();

        rtree.query( bgi::nearest( Point( x, y, z ), k ),
                     std::back_inserter( returned_values ) );
        sol = DataTransferKit::Details::nearest( bvh, {x, y, z}, k );
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
