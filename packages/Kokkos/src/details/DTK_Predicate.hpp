#ifndef DTK_PREDICATE_HPP
#define DTK_PREDICATE_HPP

namespace DataTransferKit
{
namespace Details
{

struct NearestPredicateTag
{
};
struct SpatialPredicateTag
{
};

template <typename Geometry>
struct Nearest
{
    using Tag = NearestPredicateTag;
    Nearest( Geometry const &geometry, int k )
        : _geometry( geometry )
        , _k( k )
    {
    }

    Geometry _geometry;
    int _k;
};

template <typename Geometry>
struct Within
{
    using Tag = SpatialPredicateTag;
    Within( Geometry const &geometry, double radius )
        : _geometry( geometry )
        , _radius( radius )
    {
    }

    Geometry _geometry;
    double _radius;
};

template <typename Geometry>
Nearest<Geometry> nearest( Geometry const &g, int k = 1 )
{
    return Nearest<Geometry>( g, k );
}

template <typename Geometry>
Within<Geometry> within( Geometry const &g, double r )
{
    return Within<Geometry>( g, r );
}
}
}

#endif
