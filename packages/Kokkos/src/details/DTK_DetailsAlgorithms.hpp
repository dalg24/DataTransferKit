#ifndef DTK_DETAILS_ALGORITHMS_HPP
#define DTK_DETAILS_ALGORITHMS_HPP

#include <DTK_LinearBVH.hpp> // AABB
#include <array>

namespace DataTransferKit
{
namespace Details
{
using Point = std::array<double, 3>;
using Box = AABB;

// ditance point-point
double distance( Point const &a, Point const &b );
// ditance point-box
double distance( Point const &point, Box const &box );
// expand an axis-aligned bounding dox using a point
void expand( Box &box, Point const &point );
// expand an axis-aligned bounding dox using another one
void expand( Box &box, Box const &other );
// check if two axis-aligned bounding boxes overlap
bool overlaps( Box const &box, Box const &other );

} // end namespace Details
} // end namespace DataTransferKit

#endif
