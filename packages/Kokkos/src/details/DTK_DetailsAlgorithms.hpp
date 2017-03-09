#ifndef DTK_DETAILS_ALGORITHMS_HPP
#define DTK_DETAILS_ALGORITHMS_HPP

#include <DTK_AABB.hpp>
#include <DTK_KokkosHelpers.hpp>

namespace DataTransferKit
{
namespace Details
{
using Point = Kokkos::Array<double, 3>;
using Box = AABB;

// distance point-point
double distance( Point const &a, Point const &b );
// distance point-box
double distance( Point const &point, Box const &box );
// expand an axis-aligned bounding box to include a point
void expand( Box &box, Point const &point );
// expand an axis-aligned bounding box to include another box
void expand( Box &box, Box const &other );
// check if two axis-aligned bounding boxes overlap
bool overlaps( Box const &box, Box const &other );
// calculate the centroid of a box
void centroid( Box const &box, Point &c );

namespace Functor
{
using Box = AABB;

class ExpandBoxWithBox
{
  public:
    ExpandBoxWithBox( Box const *bounding_boxes )
        : _greatest( Kokkos::ArithTraits<double>::max() )
        , _lowest( -_greatest )
        , _bounding_boxes( bounding_boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init( Box &box )
    {
        for ( int d = 0; d < 3; ++d )
        {
            box[2 * d] = _greatest;
            box[2 * d + 1] = _lowest;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i, Box &box ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            box[2 * d] =
                KokkosHelpers::min( _bounding_boxes[i][2 * d], box[2 * d] );
            box[2 * d + 1] = KokkosHelpers::max( _bounding_boxes[i][2 * d + 1],
                                                 box[2 * d + 1] );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile Box &dst, volatile Box const &src ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            // We need to access the underlying element directly because
            // operator[] is not volatile
            dst._minmax.m_elem[2 * d] = KokkosHelpers::min(
                src._minmax.m_elem[2 * d], dst._minmax.m_elem[2 * d] );
            dst._minmax.m_elem[2 * d + 1] = KokkosHelpers::max(
                src._minmax.m_elem[2 * d + 1], dst._minmax.m_elem[2 * d + 1] );
        }
    }

  private:
    double const _greatest;
    double const _lowest;
    Box const *_bounding_boxes;
};
}

} // end namespace Details
} // end namespace DataTransferKit

#endif
