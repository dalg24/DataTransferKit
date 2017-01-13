#include <details/DTK_DetailsAlgorithms.hpp>

#include <Teuchos_UnitTestHarness.hpp>

TEUCHOS_UNIT_TEST( LinearBVH, point_box_distance )
{
    namespace dtk = DataTransferKit::Details;
    // box is unit square
    std::array<double, 6> box = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    // distance is zero if the point is inside the box
    TEST_EQUALITY( dtk::distance( {0.5, 0.5, 0.5}, box ), 0.0 );
    // or anywhere on the boundary
    TEST_EQUALITY( dtk::distance( {0.0, 0.0, 0.5}, box ), 0.0 );
    // normal projection onto center of one face
    TEST_EQUALITY( dtk::distance( {2.0, 0.5, 0.5}, box ), 1.0 );
    // projection onto edge
    TEST_EQUALITY( dtk::distance( {2.0, 0.75, -1.0}, box ), std::sqrt( 2.0 ) );
    // projection onto corner node
    TEST_EQUALITY( dtk::distance( {-1.0, 2.0, 2.0}, box ), std::sqrt( 3.0 ) );
}

TEUCHOS_UNIT_TEST( LinearBVH, axis_aligned_bounding_boxes )
{
    namespace dtk = DataTransferKit::Details;
    DataTransferKit::AABB a, b;
    TEST_ASSERT( !dtk::overlaps( a, b ) );
    a._minmax = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    TEST_ASSERT( !dtk::overlaps( a, b ) );
    b._minmax = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    TEST_ASSERT( dtk::overlaps( a, b ) );
    // share same surface
    b._minmax = {1.0, 2.0, 0.0, 1.0, 0.0, 1.0};
    TEST_ASSERT( dtk::overlaps( a, b ) );
    b._minmax = {2.0, 3.0, 0.0, 1.0, 0.0, 1.0};
    TEST_ASSERT( !dtk::overlaps( a, b ) );
    b._minmax = {0.0, 1.0, 0.0, 1.0, -2.0, -1.0};
    TEST_ASSERT( !dtk::overlaps( a, b ) );
    // TODO: check union of bounding boxes
    // didn't bother testing more extensively yet because still not sure of the
    // interface

    a._minmax = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    b._minmax = {1.0, 2.0, 0.0, 1.0, 0.0, 1.0};
    dtk::expand( a, b );
}
