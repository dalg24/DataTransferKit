/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_NANOFLANN_ADAPTERS_HPP
#define DTK_NANOFLANN_ADAPTERS_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_Point.hpp>

#include <Kokkos_View.hpp>

#include <nanoflann.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
struct NanoflannPointCloudAdapter
{
    using CoordinateType = double; // NOTE: matches DTK_Point.hpp
    using SizeType = size_t;

    NanoflannPointCloudAdapter(
        Kokkos::View<Point *, DeviceType> const &points )
        : _points( points )
    {
    }

    // Must return the number of data poins
    inline SizeType kdtree_get_point_count() const { return _points.span(); }
    // Must return the dim'th component of the idx'th point in the class:
    inline CoordinateType kdtree_get_pt( const SizeType idx, int dim ) const
    {
        return _points( idx )[dim];
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data
    // point with index "idx_p2" stored in the class:
    inline double kdtree_distance( const CoordinateType *p1,
                                   const SizeType idx_p2, SizeType ) const
    {
        CoordinateType const d0 = p1[0] - _points( idx_p2 )[0];
        CoordinateType const d1 = p1[1] - _points( idx_p2 )[1];
        CoordinateType const d2 = p1[2] - _points( idx_p2 )[2];
        return d0 * d0 + d1 * d1 + d2 * d2;
        // return Details::distance( reinterpret_cast<Point const &>( *p1 ),
        //                          _points( idx_p2 ) );
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox( BBOX & ) const
    {
        return false;
    }

  private:
    Kokkos::View<Point *, DeviceType> _points;
};

} // end namespace DataTransferKit

#endif
