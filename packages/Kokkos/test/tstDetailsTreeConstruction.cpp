#include <details/DTK_DetailsTreeConstruction.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <bitset>
#include <sstream>
#include <vector>

TEUCHOS_UNIT_TEST( DetailsBVH, indirect_sort )
{
    // need a functionality that sort objects based on their Morton code and
    // also returns the indices in the original configuration

    // dummy unsorted morton codes and corresponding sorted indices as reference
    // solution
    std::vector<unsigned int> x = {2, 1, 4, 3};
    std::vector<int> ref = {1, 0, 3, 2};
    // distribute ids to unsorted objects
    int const n = x.size();
    std::vector<int> ids( n );
    std::iota( ids.begin(), ids.end(), 0 );
    // sort morton codes and object ids
    DataTransferKit::Details::sortObjects( x.data(), ids.data(), n );
    // check that they are sorted
    TEST_ASSERT( std::is_sorted( x.begin(), x.end() ) );
    // check that ids are properly ordered
    for ( int i = 0; i < n; ++i )
        TEST_EQUALITY( ids[i], ref[i] );
}

#define __clz( x ) __builtin_clz( x )

TEUCHOS_UNIT_TEST( DetailsBVH, number_of_leading_zero_bits )
{
    // this is not a proper test
    // it turns out NVDIA's implementation of int __clz(unsigned int x) is
    // slightly different than GCC __builtin_clz
    // this caused a bug in an early implementation of the function that compute
    // the common prefixes betwwen two keys (NB: when i == j)
    std::cout << "__clz\n";
    for ( int i = 0; i < 10; ++i )
        std::cout << i << "  " << std::bitset<32>( i ) << "  " << __clz( i )
                  << "\n";
    std::cout << "NOTE: when x is 0, the result of __clz(x) is undefined\n";

    std::cout << "common prefix __clz(x^y)\n";
    std::cout << " "
              << "    ";
    for ( int j = 0; j < 10; ++j )
        std::cout << " " << j << "  ";
    std::cout << "\n";
    for ( int i = 0; i < 10; ++i )
    {
        std::cout << i << "    ";
        for ( int j = 0; j < i; ++j )
            std::cout << __clz( i ^ j ) << "  ";
        std::cout << "\n";
    }
}

TEUCHOS_UNIT_TEST( DetailsBVH, example_tree_construction )
{
    // This is the example from the articles by Karras.
    // See
    // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
    std::vector<unsigned int> sorted_morton_codes;
    for ( std::string const &s : {
              "00001", "00010", "00100", "00101", "10011", "11000", "11001",
              "11110",
          } )
    {
        std::bitset<6> b( s );
        std::cout << b << "  " << b.to_ulong() << "\n";
        sorted_morton_codes.push_back( b.to_ulong() );
    }
    int const n = sorted_morton_codes.size();
    std::vector<int> sorted_indices( n );
    std::iota( sorted_indices.begin(), sorted_indices.end(), 0 );

    std::vector<DataTransferKit::LeafNode> leaf_nodes( n );
    std::vector<DataTransferKit::InternalNode> internal_nodes( n - 1 );
    DataTransferKit::Details::generateHierarchy(
        sorted_morton_codes.data(), sorted_indices.data(), n, leaf_nodes.data(),
        internal_nodes.data() );

    DataTransferKit::InternalNode *root = internal_nodes.data();
    TEST_ASSERT( root->parent == nullptr );

    std::function<void( DataTransferKit::Node *, std::ostream & )>
        traverseRecursive;
    traverseRecursive = [&leaf_nodes, &internal_nodes, &traverseRecursive](
        DataTransferKit::Node *node, std::ostream &os ) {
        if ( auto leaf = dynamic_cast<DataTransferKit::LeafNode *>( node ) )
        {
            os << "L" << leaf - leaf_nodes.data();
        }
        else
        {
            auto internal =
                dynamic_cast<DataTransferKit::InternalNode *>( node );
            os << "I" << internal - internal_nodes.data();
            traverseRecursive( internal->childA, os );
            traverseRecursive( internal->childB, os );
        }
    };
    std::ostringstream sol;
    traverseRecursive( root, sol );
    std::cout << "sol=" << sol.str() << "\n";

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

    TEST_EQUALITY( sol.str().compare( ref.str() ), 0 );
}
