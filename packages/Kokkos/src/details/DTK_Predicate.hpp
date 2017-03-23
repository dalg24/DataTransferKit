#ifndef DTK_PREDICATE_HPP
#define DTK_PREDICATE_HPP

#include <DTK_Node.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>

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

class Within
{
  public:
    using Tag = SpatialPredicateTag;
    Within( Point const &query_point, double const radius )
        : _radius( radius )
        , _query_point( query_point )
    {
    }

    bool operator()( Node const *node ) const
    {
        double node_distance = distance( _query_point, node->bounding_box );
        return ( node_distance <= _radius ) ? true : false;
    }

  private:
    double const _radius;
    Point const &_query_point;
};

template <typename Geometry>
Nearest<Geometry> nearest( Geometry const &g, int k = 1 )
{
    return Nearest<Geometry>( g, k );
}
}
}

#endif
