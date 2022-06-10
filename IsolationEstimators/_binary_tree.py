import numpy as np
from Common import get_array_module


# def fix_upper_boundaries(self, l_upper_boundaries, global_upper_boundary):
#     xp, _ = get_array_module(l_upper_boundaries)
#     l_upper_boundaries = for_list(l_upper_boundaries)
#     global_upper_boundary = for_list(global_upper_boundary)

#     l_upper_boundaries[l_upper_boundaries >= global_upper_boundary] = (
#         xp.tile(global_upper_boundary, [l_upper_boundaries.shape[0], 1])[
#             l_upper_boundaries >= global_upper_boundary
#         ]
#         + xp.finfo(xp.float32).eps
#     )
#     return l_upper_boundaries


def for_list(boundary, always_copy=True):
    xp, xpUtils = get_array_module(boundary)
    result = boundary
    if boundary.ndim == 2:
        if always_copy:
            result = xpUtils.copy(boundary)
    else:
        result = xp.expand_dims(boundary, axis=0)

    return result


def volumes(l_lower_boundaries, l_upper_boundaries):
    xp, _ = get_array_module(l_lower_boundaries)
    l_lower_boundaries = for_list(l_lower_boundaries)
    l_upper_boundaries = for_list(l_upper_boundaries)
    return xp.prod(l_upper_boundaries - l_lower_boundaries, axis=1)


def single_split(lower_boundary, upper_boundary, split_dim, split_value):
    xp, _ = get_array_module(lower_boundary)
    lower_boundary_left = for_list(lower_boundary)

    upper_boundary_left = for_list(upper_boundary)
    # upper_boundary_left[0, split_dim] = split_value
    upper_boundary_left = xp.where(
        xp.expand_dims(xp.arange(upper_boundary_left.shape[1]), axis=0) == split_dim,
        split_value,
        upper_boundary_left,
    )

    lower_boundary_right = for_list(lower_boundary)
    # lower_boundary_right[0, split_dim] = split_value
    lower_boundary_right = xp.where(
        xp.expand_dims(xp.arange(lower_boundary_right.shape[1]), axis=0) == split_dim,
        split_value,
        lower_boundary_right,
    )

    upper_boundary_right = for_list(upper_boundary)

    if np.any(lower_boundary_left == upper_boundary_left) or np.any(
        lower_boundary_right == upper_boundary_right
    ):
        print("wth")
    return (
        lower_boundary_left,
        upper_boundary_left,
        lower_boundary_right,
        upper_boundary_right,
    )


def brother(l_parents, i_node):
    xp, _ = get_array_module(l_parents)
    l_brothers = xp.where(l_parents == l_parents[i_node])[0]
    return l_brothers[l_brothers != i_node][0]


NO_PARENT = -1
NO_SPLIT = -1
ROOT_LEVEL = 0


class AxisParallelBinaryTree:
    def seed(self, lower_boundary, upper_boundary):
        xp, _ = get_array_module(lower_boundary)
        self.node_parents = xp.array([NO_PARENT], dtype=int)
        self.node_split_dim = xp.array([NO_SPLIT], dtype=int)
        self.node_split_value = xp.array([NO_SPLIT], dtype=float)
        self.node_levels = xp.array([ROOT_LEVEL], dtype=int)

        self.node_lower_boundaries = for_list(lower_boundary)
        self.node_upper_boundaries = for_list(upper_boundary)

        self.global_lower_boundary = for_list(lower_boundary)  # constant
        self.global_upper_boundary = for_list(upper_boundary)  # constant
        return

    def grow(self, i_node, split_dim, split_value):
        xp, _ = get_array_module(self.node_parents)

        if self.node_split_dim[i_node] != NO_SPLIT:
            raise Exception("Grow operation must be on a leaf node!")

        i_left = self.node_parents.shape[0]
        i_right = i_left + 1

        n_node = self.node_split_dim.shape[0]

        # self.node_split_dim[i_node] = split_dim
        self.node_split_dim = xp.where(
            xp.arange(n_node) == i_node, split_dim, self.node_split_dim
        )
        self.node_split_dim = xp.concatenate(
            [self.node_split_dim, xp.array([NO_SPLIT, NO_SPLIT], dtype=int)], axis=0
        )

        # self.node_split_value[i_node] = split_value
        self.node_split_value = xp.where(
            xp.arange(n_node) == i_node, split_value, self.node_split_value
        )
        self.node_split_value = xp.concatenate(
            [self.node_split_value, xp.array([NO_SPLIT, NO_SPLIT], dtype=float)],
            axis=0,
        )

        (
            lower_boundary_left,
            upper_boundary_left,
            lower_boundary_right,
            upper_boundary_right,
        ) = single_split(
            self.node_lower_boundaries[i_node : i_node + 1, :],
            self.node_upper_boundaries[i_node : i_node + 1, :],
            split_dim,
            split_value,
        )
        self.node_lower_boundaries = xp.concatenate(
            [self.node_lower_boundaries, lower_boundary_left, lower_boundary_right],
            axis=0,
        )
        self.node_upper_boundaries = xp.concatenate(
            [self.node_upper_boundaries, upper_boundary_left, upper_boundary_right],
            axis=0,
        )

        self.node_parents = xp.concatenate(
            [self.node_parents, xp.array([i_node, i_node], dtype=int)], axis=0
        )

        new_level = self.node_levels[i_node] + 1
        self.node_levels = xp.concatenate(
            [self.node_levels, xp.array([new_level, new_level])], axis=0
        )

        return i_left, i_right

    def prune(self, i_node):
        xp, xpUtils = get_array_module(self.node_parents)

        if self.node_split_dim[i_node] != NO_SPLIT:
            raise Exception("Prune operation must be on a leaf node!")

        l_volumes = volumes(self.node_lower_boundaries, self.node_upper_boundaries)

        i_parent = self.node_parents[i_node]
        i_brother = brother(self.node_parents, i_node)

        self._single_merge(
            self.node_lower_boundaries[i_node : i_node + 1, :],
            self.node_upper_boundaries[i_node : i_node + 1, :],
            i_brother,
        )

        l_volumes_new = volumes(self.node_lower_boundaries, self.node_upper_boundaries)
        l_volumes_diff = l_volumes_new - l_volumes

        node_count = self.node_parents.shape[0]
        l_idx = xp.arange(node_count)
        l_keep = xp.logical_and(l_idx != i_node, l_idx != i_brother)

        # print({"me": i_node, "brother": i_brother, "parent": self.node_parents[i_node]})
        # print(xp.arange(node_count))

        # print(self.node_parents)

        l_nephew = self.node_parents == i_brother
        if xp.any(l_nephew):
            # Let the parent node adopt brother's children
            # self.node_parents[self.node_parents == i_brother] = self.node_parents[i_node]
            self.node_parents = xp.where(
                self.node_parents == i_brother,
                self.node_parents[i_node],
                self.node_parents,
            )
            # print(self.node_parents)

        # update parent split to brother split
        # self.node_split_dim[i_parent] = self.node_split_dim[i_brother]
        self.node_split_dim = xp.where(
            xp.arange(self.node_split_dim.shape[0]) == i_parent,
            self.node_split_dim[i_brother],
            self.node_split_dim,
        )
        # self.node_split_value[i_parent] = self.node_split_value[i_brother]
        self.node_split_value = xp.where(
            xp.arange(self.node_split_value.shape[0]) == i_parent,
            self.node_split_value[i_brother],
            self.node_split_value,
        )

        # remove the node and brother
        self.node_split_dim = self.node_split_dim[l_keep]
        self.node_split_value = self.node_split_value[l_keep]

        self.node_levels = self.node_levels[l_keep]

        self.node_parents = self.node_parents[l_keep]
        # print(self.node_parents)
        # and then update the parents to new index
        # l_idx_update = l_idx.copy()
        # l_idx_update[l_keep] = xp.arange(node_count - 2)
        l_idx_update = xpUtils.tensor_scatter_nd_update(
            l_idx, xp.where(l_keep)[0], xp.arange(node_count - 2)
        )

        # self.node_parents[self.node_parents >= 0] = l_idx_update[self.node_parents][
        #     self.node_parents >= 0
        # ]
        self.node_parents = xp.where(
            self.node_parents >= 0,
            xp.take(l_idx_update, self.node_parents),
            self.node_parents,
        )
        # print(self.node_parents)

        self.node_lower_boundaries = xp.take(
            self.node_lower_boundaries, xp.where(l_keep)[0], axis=0
        )
        self.node_upper_boundaries = xp.take(
            self.node_upper_boundaries, xp.where(l_keep)[0], axis=0
        )

        l_volumes_diff = l_volumes_diff[l_keep]

        # TODO: remove the following before return
        l = self.leaf()
        v = self.volumes(l)
        if xp.sum(v) < 0.99:
            print(l)
            print(v)
            print("WTF")
        return l_volumes_diff, l_keep

    def _single_merge(self, lower_boundary, upper_boundary, i_node):
        xp, _ = get_array_module(self.node_parents)

        lower_boundary_new = for_list(lower_boundary)
        lower_boundary_old = self.node_lower_boundaries[i_node : i_node + 1, :]
        # lower_to_replace = lower_boundary_old < lower_boundary_new
        # lower_boundary_new[lower_to_replace] = lower_boundary_old[lower_to_replace]
        lower_boundary_new = xp.where(
            lower_boundary_old < lower_boundary_new,
            lower_boundary_old,
            lower_boundary_new,
        )
        # self.node_lower_boundaries[i_node : i_node + 1, :] = lower_boundary_new
        self.node_lower_boundaries = xp.where(
            xp.expand_dims(xp.arange(self.node_lower_boundaries.shape[0]), axis=1)
            == i_node,
            lower_boundary_new,
            self.node_lower_boundaries,
        )

        upper_boundary_new = for_list(upper_boundary)
        upper_boundary_old = self.node_upper_boundaries[i_node : i_node + 1, :]
        # upper_to_replace = upper_boundary_old > upper_boundary_new
        # upper_boundary_new[upper_to_replace] = upper_boundary_old[upper_to_replace]
        upper_boundary_new = xp.where(
            upper_boundary_old > upper_boundary_new,
            upper_boundary_old,
            upper_boundary_new,
        )
        # self.node_upper_boundaries[i_node : i_node + 1, :] = upper_boundary_new
        self.node_upper_boundaries = xp.where(
            xp.expand_dims(xp.arange(self.node_upper_boundaries.shape[0]), axis=1)
            == i_node,
            upper_boundary_new,
            self.node_upper_boundaries,
        )

        l_children = xp.where(self.node_parents == i_node)[0]

        if l_children.shape[0] == 2:
            split_dim = self.node_split_dim[i_node]
            split_value = self.node_split_value[i_node]

            (
                lower_boundary_left,
                upper_boundary_left,
                lower_boundary_right,
                upper_boundary_right,
            ) = single_split(
                for_list(lower_boundary),
                for_list(upper_boundary),
                split_dim,
                split_value,
            )

            i_left = l_children[0]
            self._single_merge(lower_boundary_left, upper_boundary_left, i_left)
            # self.node_levels[i_left] = self.node_levels[i_left] - 1
            self.node_levels = xp.where(
                xp.arange(self.node_levels.shape[0]) == i_left,
                self.node_levels[i_left] - 1,
                self.node_levels,
            )

            i_right = l_children[1]
            self._single_merge(lower_boundary_right, upper_boundary_right, i_right)
            # self.node_levels[i_right] = self.node_levels[i_right] - 1
            self.node_levels = xp.where(
                xp.arange(self.node_levels.shape[0]) == i_right,
                self.node_levels[i_left] - 1,
                self.node_levels,
            )
        # TODO: remove the rest
        elif l_children.shape[0] == 0:
            return
        else:
            print("WTH")

    def search(self, X, l_nodes=None):
        xp, _ = get_array_module(X)
        if l_nodes is None:
            l_lower_boundaries = self.node_lower_boundaries
            l_upper_boundaries = self.node_upper_boundaries
        else:
            # l_lower_boundaries = self.node_lower_boundaries[l_nodes, :]
            l_lower_boundaries = xp.take(
                self.node_lower_boundaries, xp.where(l_nodes)[0], axis=0
            )
            # l_upper_boundaries = self.node_upper_boundaries[l_nodes, :]
            l_upper_boundaries = xp.take(
                self.node_upper_boundaries, xp.where(l_nodes)[0], axis=0
            )
        return self._search(X, l_lower_boundaries, l_upper_boundaries)

    def _search(self, X, l_lower_boundaries, l_upper_boundaries):
        xp, xpUtils = get_array_module(X)
        l_lower_boundaries = for_list(l_lower_boundaries)
        l_upper_boundaries = for_list(l_upper_boundaries)
        # l_upper_boundaries[l_upper_boundaries >= self.global_upper_boundary] = (
        #     xpUtils.tile(self.global_upper_boundary, [l_upper_boundaries.shape[0], 1])[
        #         l_upper_boundaries >= self.global_upper_boundary
        #     ]
        #     + xp.finfo(xp.float32).eps
        # )

        l_upper_boundaries = xp.where(
            l_upper_boundaries >= self.global_upper_boundary,
            xpUtils.tile(self.global_upper_boundary, [l_upper_boundaries.shape[0], 1])
            + xp.finfo(xp.float32).eps,
            l_upper_boundaries,
        )

        result = xp.logical_and(
            xp.all(
                xp.expand_dims(X, axis=1) >= xp.expand_dims(l_lower_boundaries, axis=0),
                axis=2,
            ),
            xp.all(
                xp.expand_dims(X, axis=1) < xp.expand_dims(l_upper_boundaries, axis=0),
                axis=2,
            ),
        )  # n_x * n_nodes

        return result

    def volumes(self, l_nodes=None):
        if l_nodes is None:
            l_lower_boundaries = self.node_lower_boundaries
            l_upper_boundaries = self.node_upper_boundaries
        else:
            xp, _ = get_array_module(self.node_lower_boundaries)
            # l_lower_boundaries = self.node_lower_boundaries[l_nodes, :]
            l_lower_boundaries = xp.take(
                self.node_lower_boundaries, xp.where(l_nodes)[0], axis=0
            )
            # l_upper_boundaries = self.node_upper_boundaries[l_nodes, :]
            l_upper_boundaries = xp.take(
                self.node_upper_boundaries, xp.where(l_nodes)[0], axis=0
            )
        return volumes(l_lower_boundaries, l_upper_boundaries)

    def leaf(self, return_boolean=True):
        if return_boolean:
            return self.node_split_dim == NO_SPLIT
        else:
            xp, _ = get_array_module(self.node_split_dim)
            return xp.where(self.node_split_dim == NO_SPLIT)[0]
