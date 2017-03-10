#include <DTK_TreeConstruction.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>

#include <Kokkos_ArithTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <bitset>
#include <sstream>
#include <vector>

namespace dtk = DataTransferKit::Details;

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( DetailsBVH, morton_codes, SC, LO, GO, NO )
{
    std::vector<dtk::Point> points = {
        dtk::Point( {0.0, 0.0, 0.0} ),
        dtk::Point( {0.25, 0.75, 0.25} ),
        dtk::Point( {0.75, 0.25, 0.25} ),
        dtk::Point( {0.75, 0.75, 0.25} ),
        dtk::Point( {1.33, 2.33, 3.33} ),
        dtk::Point( {1.66, 2.66, 3.66} ),
        dtk::Point( {1024.0, 1024.0, 1024.0} ),
    };
    int const n = points.size();
    // lower left front corner corner of the octant the points fall in
    std::vector<std::array<unsigned int, 3>> anchors = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0},         {0, 0, 0},
        {1, 2, 3}, {1, 2, 3}, {1023, 1023, 1023}};
    auto fun = []( std::array<unsigned int, 3> const &anchor ) {
        unsigned int i = std::get<0>( anchor );
        unsigned int j = std::get<1>( anchor );
        unsigned int k = std::get<2>( anchor );
        return 4 * dtk::expandBits( i ) + 2 * dtk::expandBits( j ) +
               dtk::expandBits( k );
    };
    std::vector<unsigned int> ref( n,
                                   Kokkos::ArithTraits<unsigned int>::max() );
    for ( int i = 0; i < n; ++i )
        ref[i] = fun( anchors[i] );
    // using points rather than boxes for convenience here but still have to
    // build the axis-aligned bounding boxes around them
    std::vector<dtk::Box> boxes( n );
    for ( int i = 0; i < n; ++i )
        dtk::expand( boxes[i], points[i] );

    dtk::Box scene;
    using DeviceType = typename NO::device_type;
    DataTransferKit::TreeConstruction<SC, LO, GO, NO> tc;
    tc.calculateBoundingBoxOfTheScene( boxes.data(), n, scene );
    for ( int d = 0; d < 3; ++d )
    {
        TEST_EQUALITY( scene[2 * d + 0], 0.0 );
        TEST_EQUALITY( scene[2 * d + 1], 1024.0 );
    }

    Kokkos::View<unsigned int *, DeviceType> morton_codes( "morton_codes", n );
    for ( int i = 0; i < n; ++i )
        morton_codes[i] = Kokkos::ArithTraits<unsigned int>::max();
    tc.assignMortonCodes( boxes.data(), morton_codes, n, scene );
    TEST_COMPARE_ARRAYS( morton_codes, ref );
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( DetailsBVH, indirect_sort, SC, LO, GO, NO )
{
    // need a functionality that sort objects based on their Morton code and
    // also returns the indices in the original configuration

    // dummy unsorted Morton codes and corresponding sorted indices as reference
    // solution
    //
    using DeviceType = typename NO::device_type;
    int constexpr n = 4;
    Kokkos::View<unsigned int *, DeviceType> k( "k", n );
    k[0] = 2;
    k[1] = 1;
    k[2] = 4;
    k[3] = 3;
    std::vector<int> ref = {1, 0, 3, 2};
    // distribute ids to unsorted objects
    Kokkos::View<int *, DeviceType> ids( "ids", n );
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    // sort morton codes and object ids
    DataTransferKit::TreeConstruction<SC, LO, GO, NO> tc;
    tc.sortObjects( k, ids, n );
    // check that they are sorted
    TEST_ASSERT( std::is_sorted( k.data(), k.data() + n ) );
    // check that ids are properly ordered
    TEST_COMPARE_ARRAYS( ids, ref );
}

TEUCHOS_UNIT_TEST( DetailsBVH, number_of_leading_zero_bits )
{
    TEST_EQUALITY( dtk::countLeadingZeros( 0 ), 32 );
    TEST_EQUALITY( dtk::countLeadingZeros( 1 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 5 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 6 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 7 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 8 ), 28 );
    TEST_EQUALITY( dtk::countLeadingZeros( 9 ), 28 );
    // bitwise exclusive OR operator to compare bits
    TEST_EQUALITY( dtk::countLeadingZeros( 1 ^ 0 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ^ 0 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ^ 1 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 0 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 1 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 2 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 0 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 1 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 2 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 3 ), 29 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( DetailsBVH, common_prefix, SC, LO, GO, NO )
{
    using DeviceType = typename NO::device_type;
    int const n = 13;
    // NOTE: Morton codes below are **not** unique
    Kokkos::View<unsigned int *, DeviceType> fi( "fi", n );
    fi[0] = 0;
    fi[1] = 1;
    fi[2] = 1;
    fi[3] = 2;
    fi[4] = 3;
    fi[5] = 5;
    fi[6] = 8;
    fi[7] = 13;
    fi[8] = 21;
    fi[9] = 34;
    fi[10] = 55;
    fi[11] = 89;
    fi[12] = 144;

    DataTransferKit::TreeConstruction<SC, LO, GO, NO> tc;
    TEST_EQUALITY( tc.commonPrefix( fi, n, 0, 0 ), 32 + 32 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 0, 1 ), 31 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 1, 0 ), 31 );
    // duplicate Morton codes
    TEST_EQUALITY( fi[1], 1 );
    TEST_EQUALITY( fi[1], fi[2] );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 1, 1 ), 64 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 1, 2 ), 32 + 30 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 2, 1 ), 62 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 2, 2 ), 64 );
    // by definition \delta(i, j) = -1 when j \notin [0, n-1]
    TEST_EQUALITY( tc.commonPrefix( fi, n, 0, -1 ), -1 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 12, 12 ), 64 );
    TEST_EQUALITY( tc.commonPrefix( fi, n, 12, 13 ), -1 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( DetailsBVH, example_tree_construction, SC,
                                   LO, GO, NO )
{
    // This is the example from the articles by Karras.
    // See
    // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
    using DeviceType = typename NO::device_type;
    int const n = 8;
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes(
        "sorted_morton_codes", n );
    std::vector<std::string> s{
        "00001", "00010", "00100", "00101", "10011", "11000", "11001", "11110",
    };
    for ( int i = 0; i < n; ++i )
    {
        std::bitset<6> b( s[i] );
        std::cout << b << "  " << b.to_ulong() << "\n";
        sorted_morton_codes[i] = b.to_ulong();
    }

    // reference solution for a recursive traversal from top to bottom
    // starting from root, visiting first the left child and then the right one
    std::ostringstream ref;
    ref << "I0"
        << "I3"
        << "I1"
        << "L0"
        << "L1"
        << "I2"
        << "L2"
        << "L3"
        << "I4"
        << "L4"
        << "I5"
        << "I6"
        << "L5"
        << "L6"
        << "L7";
    std::cout << "ref=" << ref.str() << "\n";

    // hierarchy generation
    Kokkos::View<DataTransferKit::Node *, DeviceType> leaf_nodes( "leaf_nodes",
                                                                  n );
    Kokkos::View<DataTransferKit::Node *, DeviceType> internal_nodes(
        "internal_nodes", n - 1 );
    std::function<void( DataTransferKit::Node *, std::ostream & )>
        traverseRecursive;
    traverseRecursive = [&leaf_nodes, &internal_nodes, &traverseRecursive](
        DataTransferKit::Node *node, std::ostream &os ) {
        if ( std::any_of( leaf_nodes.data(), leaf_nodes.data() + n,
                          [node]( DataTransferKit::Node const &leaf_node ) {
                              return std::addressof( leaf_node ) == node;
                          } ) )
        {
            os << "L" << node - leaf_nodes.data();
        }
        else
        {
            os << "I" << node - internal_nodes.data();
            for ( DataTransferKit::Node *child :
                  {node->children.first, node->children.second} )
                traverseRecursive( child, os );
        }
    };

    DataTransferKit::TreeConstruction<SC, LO, GO, NO> tc;
    tc.generateHierarchy( sorted_morton_codes, n, leaf_nodes, internal_nodes );

    DataTransferKit::Node *root = internal_nodes.data();
    TEST_ASSERT( root->parent == nullptr );

    std::ostringstream sol;
    traverseRecursive( root, sol );
    std::cout << "sol=" << sol.str() << "\n";

    TEST_EQUALITY( sol.str().compare( ref.str() ), 0 );
}

// Include the test macros.
#include "DataTransferKitKokkos_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( SCALAR, LO, GO, NODE )                                \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( DetailsBVH, morton_codes, SCALAR,    \
                                          LO, GO, NODE )                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( DetailsBVH, indirect_sort, SCALAR,   \
                                          LO, GO, NODE )                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( DetailsBVH, common_prefix, SCALAR,   \
                                          LO, GO, NODE )                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(                                      \
        DetailsBVH, example_tree_construction, SCALAR, LO, GO, NODE )
// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_SLGN( UNIT_TEST_GROUP )
