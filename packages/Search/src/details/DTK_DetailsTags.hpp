/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_DETAILS_TAGS_HPP
#define DTK_DETAILS_TAGS_HPP

#include <DTK_Box.hpp>
#include <DTK_Point.hpp>

namespace DataTransferKit
{
namespace Details
{

struct PointTag
{
};

struct BoxTag
{
};

template <typename Geometry>
struct Tag
{
};

template <>
struct Tag<Point>
{
    using type = PointTag;
};

template <>
struct Tag<Box>
{
    using type = BoxTag;
};

} // namespace Details
} // namespace DataTransferKit

#endif
