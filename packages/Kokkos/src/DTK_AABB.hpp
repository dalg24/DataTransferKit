#ifndef DTK_AABB_HPP
#define DTK_AABB_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

namespace DataTransferKit
{
// Axis-Aligned Bounding Box
// This is just a thin wrapper around an array of size 2x spatial dimension with
// a default constructor to initialize properly an "empty" box.
struct AABB
{
    using ArrayType = Kokkos::Array<double, 6>;
    using SizeType = ArrayType::size_type;
    AABB() = default;
    AABB( ArrayType const &minmax )
        : _minmax( minmax )
    {
    }
    AABB &operator=( ArrayType const &minmax )
    {
        _minmax = minmax;
        return *this;
    }
    double &operator[]( SizeType i ) { return _minmax[i]; }
    double const &operator[]( SizeType i ) const { return _minmax[i]; }
    ArrayType _minmax = {{
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
        Kokkos::ArithTraits<double>::max(), -Kokkos::ArithTraits<double>::max(),
    }};
    friend std::ostream &operator<<( std::ostream &os, AABB const &aabb )
    {
        os << "{";
        for ( int d = 0; d < 3; ++d )
            os << " [" << aabb[2 * d + 0] << ", " << aabb[2 * d + 1] << "],";
        os << "}";
        return os;
    }
};
}

#endif
