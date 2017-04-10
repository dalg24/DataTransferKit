#ifndef DTK_NODE_HPP
#define DTK_NODE_HPP

#include <DTK_Box.hpp>

namespace DataTransferKit
{
struct Node
{
    Node *parent = nullptr;
    Node *children_a = nullptr;
    Node *children_b = nullptr;
    BBox bounding_box;
};
}

#endif
