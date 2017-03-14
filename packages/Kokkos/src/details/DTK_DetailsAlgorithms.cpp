#include <details/DTK_DetailsAlgorithms.hpp>

#include <cmath>

namespace DataTransferKit
{
namespace Details
{

double distance( Point const &a, Point const &b )
{
    double distance_squared = 0.0;
    for ( int d = 0; d < 3; ++d )
    {
        double tmp = b[d] - a[d];
        distance_squared += tmp * tmp;
    }
    return std::sqrt( distance_squared );
}

double distance( Point const &point, Box const &box )
{
    Point projected_point;
    for ( int d = 0; d < 3; ++d )
    {
        if ( point[d] < box[2 * d + 0] )
            projected_point[d] = box[2 * d + 0];
        else if ( point[d] > box[2 * d + 1] )
            projected_point[d] = box[2 * d + 1];
        else
            projected_point[d] = point[d];
    }
    return distance( point, projected_point );
}

void expand( Box &box, Point const &point )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( point[d] < box[2 * d + 0] )
            box[2 * d + 0] = point[d];
        if ( point[d] > box[2 * d + 1] )
            box[2 * d + 1] = point[d];
    }
}

void expand( Box &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( box[2 * d + 0] > other[2 * d + 0] )
            box[2 * d + 0] = other[2 * d + 0];
        if ( box[2 * d + 1] < other[2 * d + 1] )
            box[2 * d + 1] = other[2 * d + 1];
    }
}

bool overlaps( Box const &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
        if ( box[2 * d + 0] > other[2 * d + 1] ||
             box[2 * d + 1] < other[2 * d + 0] )
            return false;
    return true;
}

void centroid( Box const &box, Point &c )
{
    for ( int d = 0; d < 3; ++d )
        c[d] = 0.5 * ( box[2 * d + 0] + box[2 * d + 1] );
}
unsigned int expandBits( unsigned int v )
{
    v = ( v * 0x00010001u ) & 0xFF0000FFu;
    v = ( v * 0x00000101u ) & 0x0F00F00Fu;
    v = ( v * 0x00000011u ) & 0xC30C30C3u;
    v = ( v * 0x00000005u ) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D( double x, double y, double z )
{
    using std::min;
    using std::max;
    x = min( max( x * 1024.0, 0.0 ), 1023.0 );
    y = min( max( y * 1024.0, 0.0 ), 1023.0 );
    z = min( max( z * 1024.0, 0.0 ), 1023.0 );
    unsigned int xx = expandBits( (unsigned int)x );
    unsigned int yy = expandBits( (unsigned int)y );
    unsigned int zz = expandBits( (unsigned int)z );
    return xx * 4 + yy * 2 + zz;
}

// TODO: this is a mess
// we need a default impl
//#define __clz( x ) __builtin_clz( x )
// default implementation if nothing else is available
// Taken from:
// http://stackoverflow.com/questions/23856596/counting-leading-zeros-in-a-32-bit-unsigned-integer-with-best-algorithm-in-c-pro
// WARNING: this implementation does **not** support __clz(0) (result should be
// 32 but this function returns 0)
int clz( uint32_t x )
{
    static const char debruijn32[32] = {
        0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
        1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return debruijn32[x * 0x076be629 >> 27];
}

// TODO: use preprocessor directive to select an implementation
// it turns out NVDIA's implementation of int __clz(unsigned int x) is
// slightly different than GCC __builtin_clz
// this caused a bug in an early implementation of the function that compute
// the common prefixes between two keys (NB: when i == j)
int countLeadingZeros( unsigned int x )
{
#if defined __CUDACC__
    // intrinsic function that is only supported in device code
    // COMMENT: not sure how I am supposed to use it then...
    return __clz( x );

#elif defined __GNUC__
    // int __builtin_clz(unsigned int x) result is undefined if x is 0
    return x != 0 ? __builtin_clz( x ) : 32;
#else
    // similar problem with the default implementation
    return x != 0 ? clz( x ) : 32;
#endif
}

} // end namespace Details
} // end namespace DataTransferKit
