/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_NANOFLANN_KDTREE_HELPERS_HPP
#define DTK_NANOFLANN_KDTREE_HELPERS_HPP

#include <DTK_Point.hpp>
#include <DTK_Predicates.hpp>

#include "DTK_NanoflannAdapters.hpp"

#include <nanoflann.hpp>

namespace mystd
{
template <typename T, typename... Args>
std::unique_ptr<T> make_unique( Args &&... args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>( args )... ) );
}
} // namespace mystd

template <typename DeviceType>
struct NanoflannKDTreeHelpers
{
    struct KDTreeWrap
    {
        using DatasetAdapter =
            DataTransferKit::NanoflannPointCloudAdapter<DeviceType>;
        using DistanceType =
            nanoflann::L2_Simple_Adaptor<double, DatasetAdapter>;
        using KDTreeAdapter =
            nanoflann::KDTreeSingleIndexAdaptor<DistanceType, DatasetAdapter, 3,
                                                size_t>;
        KDTreeWrap(
            Kokkos::View<DataTransferKit::Point *, DeviceType> const &points )
            : _dataset( points )
            , _kdtree( 3, _dataset )
        {
            _kdtree.buildIndex();
        }
        DatasetAdapter _dataset;
        KDTreeAdapter _kdtree;
    };

    using KDTreeUniquePtr = std::unique_ptr<KDTreeWrap>;
    static KDTreeUniquePtr makeKDTree(
        Kokkos::View<DataTransferKit::Point *, DeviceType> const &points )
    {
        return mystd::make_unique<KDTreeWrap>( points );
    }

    // knnSearch
    static std::tuple<Kokkos::View<int *, DeviceType>,
                      Kokkos::View<int *, DeviceType>>
    performQueries(
        KDTreeUniquePtr const &wrap,
        Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *,
                     DeviceType>
            queries )
    {
        auto const n_queries = queries.extent_int( 0 );
        Kokkos::View<int *, DeviceType> offset( "offset", n_queries + 1 );
        std::vector<std::pair<size_t, double>> returned_indices_distances;
        for ( int i = 0; i < n_queries; ++i )
        {
            size_t const k = queries( i )._k;
            double const *const query_point = queries( i )._geometry._coords;
            std::vector<size_t> indices( k );
            std::vector<double> distances( k );
            offset( i ) = wrap->_kdtree.knnSearch(
                query_point, k, indices.data(), distances.data() );
            for ( int j = 0; j < offset( i ); ++j )
                returned_indices_distances.push_back(
                    std::make_pair( indices[j], distances[j] ) );
        }
        DataTransferKit::exclusivePrefixSum( offset );
        auto const n_results = DataTransferKit::lastElement( offset );
        Kokkos::View<int *, DeviceType> indices( "indices", n_results );
        for ( int i = 0; i < n_queries; ++i )
            for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                indices( j ) = returned_indices_distances[j].first;
        return std::make_tuple( offset, indices );
    }

    // radiusSearch
    static std::tuple<Kokkos::View<int *, DeviceType>,
                      Kokkos::View<int *, DeviceType>>
    performQueries(
        KDTreeUniquePtr const &wrap,
        Kokkos::View<DataTransferKit::Intersects<DataTransferKit::Sphere> *,
                     DeviceType>
            queries )
    {
        auto const n_queries = queries.extent_int( 0 );
        Kokkos::View<int *, DeviceType> offset( "offset", n_queries + 1 );
        std::vector<std::pair<size_t, double>> returned_indices_distances;
        for ( int i = 0; i < n_queries; ++i )
        {
            auto const sphere = queries( i )._geometry;
            double const radius = sphere.radius();
            double const *const centroid = sphere.centroid()._coords;
            std::vector<std::pair<size_t, double>> ret_matches;
            nanoflann::SearchParams params;
            offset( i ) = wrap->_kdtree.radiusSearch( centroid, radius * radius,
                                                      ret_matches, params );
            returned_indices_distances.insert( returned_indices_distances.end(),
                                               ret_matches.begin(),
                                               ret_matches.end() );
        }
        DataTransferKit::exclusivePrefixSum( offset );
        auto const n_results = DataTransferKit::lastElement( offset );
        Kokkos::View<int *, DeviceType> indices( "indices", n_results );
        for ( int i = 0; i < n_queries; ++i )
            for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                indices( j ) = returned_indices_distances[j].first;
        return std::make_tuple( offset, indices );
    }
};

#endif
