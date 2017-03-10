#ifndef DTK_NODE_HPP
#define DTK_NODE_HPP

#include <DTK_AABB.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
struct Node
{
    virtual ~Node() = default;
    Node *parent = nullptr;
    Kokkos::pair<Node *, Node *> children = {nullptr, nullptr};
    AABB bounding_box;
};
}

#endif
