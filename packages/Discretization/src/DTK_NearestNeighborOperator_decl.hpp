/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_NEAREST_NEIGHBOR_OPERATOR_DECL_HPP
#define DTK_NEAREST_NEIGHBOR_OPERATOR_DECL_HPP

#include <DTK_ConfigDefs.hpp>

#include <Kokkos_Core.hpp>
#include <Teuchos_Comm.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class NearestNeighborOperator
{
    using ExecutionSpace = typename DeviceType::execution_space;

  public:
    NearestNeighborOperator(
        Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
        Kokkos::View<Coordinate **, DeviceType> source_points,
        Kokkos::View<Coordinate **, DeviceType> target_points );

    // Cannot use Kokkos lambda in constructor:
    //   error: An extended __host__ __device__ lambda cannot be defined within
    //   a function whose address cannot be taken
    // Cannot make setup() private:
    //   error: An extended __host__ __device__ lambda cannot be defined in a
    //   member function that has private or protected access within its class
    void setup( Kokkos::View<Coordinate **, DeviceType> source_points,
                Kokkos::View<Coordinate **, DeviceType> target_points );

    void apply( Kokkos::View<double *, DeviceType> source_values,
                Kokkos::View<double *, DeviceType> target_points );

  private:
    Teuchos::RCP<const Teuchos::Comm<int>> _comm;
    Kokkos::View<int *, DeviceType> _indices;
    Kokkos::View<int *, DeviceType> _ranks;
};

} // end namespace DataTransferKit

#endif
