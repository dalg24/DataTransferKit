#include <details/DTK_DetailsAlgorithms.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{
namespace Details
{

bool checkOverlap( AABB const &a, AABB const &b ) { return overlaps( a, b ); }

} // end namespace Details
} // end namespace DataTransferKit
