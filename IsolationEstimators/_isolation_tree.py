import numpy
from sklearn.base import TransformerMixin, DensityMixin
from Common import (
    ReservoirSamplingEstimator,
    get_array_module,
    rotate,
    is_check_and_warn_enabled,
    check_and_warn,
    checked_and_warn,
)
from ._binary_tree import AxisParallelBinaryTree


def get_boundaries(X, ball_scaled=True):
    xp, _ = get_array_module(X)

    if ball_scaled:
        global_upper_boundary = xp.ones([1, X.shape[1]], dtype=X.dtype)
        global_lower_boundary = 0 - global_upper_boundary
    else:
        global_lower_boundary = X.min(axis=0, keepdims=True)
        global_upper_boundary = X.max(axis=0, keepdims=True)

    return global_lower_boundary, global_upper_boundary


class IsolationTree(
    ReservoirSamplingEstimator, AxisParallelBinaryTree, TransformerMixin
):
    def __init__(self, psi, rotation=False, global_boundaries=None, **kwargs):
        super().__init__(psi)
        self.rotation = rotation
        self.global_boundaries = global_boundaries
        return

    def fit(self, X, y=None):
        if self.rotation:
            X_, SO = rotate(X)
            self.SO_ = SO
        else:
            X_ = X
        super().fit(X, y)
        if self.global_boundaries is not None:
            global_lower_boundary, global_upper_boundary = self.global_boundaries
        else:
            global_lower_boundary, global_upper_boundary = get_boundaries(
                X_, self.rotation
            )
        super().seed(global_lower_boundary, global_upper_boundary)
        self._build_tree()
        return self

    def _isolation_split(self, lower_boundary, upper_boundary):
        xp, xpUtils = get_array_module(lower_boundary)

        l_in = self._search(self.samples_, lower_boundary, upper_boundary)
        l_in = l_in[:, 0]  # only one node therefore one column
        if xp.sum(l_in) == 1:
            return True, None, None, l_in
        elif xp.sum(l_in) < 1:
            raise Exception("There is no sample in this region!")

        m_in = xp.take(self.samples_, xp.where(l_in)[0], axis=0)

        l_dims = xp.where(xp.not_equal(xp.min(m_in, axis=0), xp.max(m_in, axis=0)))[0]
        split_dim = xp.random.randint(l_dims.shape[0])
        split_dim = l_dims[split_dim]

        sample_values = xpUtils.unique(m_in[:, split_dim])
        sample_values = xp.sort(sample_values)
        split_pos = xp.random.randint(sample_values.shape[0] - 1)
        split_value = (sample_values[split_pos] + sample_values[split_pos + 1]) / 2

        return False, split_dim, split_value, l_in

    def _build_tree(self):
        stack_to_split = [0]

        while len(stack_to_split) > 0:
            i = stack_to_split.pop()
            lower_boundary_i = self.node_lower_boundaries[i]
            upper_boundary_i = self.node_upper_boundaries[i]

            is_leaf, split_dim, split_value, _ = self._isolation_split(
                lower_boundary_i, upper_boundary_i
            )
            if not is_leaf:
                count_nodes = self.node_parents.shape[0]
                self.grow(i, split_dim, split_value)
                stack_to_split.append(count_nodes)
                stack_to_split.append(count_nodes + 1)

        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

    def partial_fit(self, X, y=None):
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, _ = rotate(X, self.SO_)
        else:
            X_ = X
        changed_count, new_samples, drop_samples, reservoir = super().update_samples(X_)

        for i in range(changed_count):
            drop_sample = drop_samples[i : i + 1, :]
            self._prune(drop_sample)
            # TODO
            # print(xp.sum(self.node_volumes_[self.node_is_leaf_]))
            # print(xp.sum(self.node_mass_[self.node_is_leaf_]))

            new_sample = new_samples[i : i + 1, :]
            self._grow(new_sample)
            # TODO
            # print(xp.sum(self.node_volumes_[self.node_is_leaf_]))
            # print(xp.sum(self.node_mass_[self.node_is_leaf_]))

        self.node_mass_ = self.node_mass_ + xp.sum(self.search(X_), axis=0, dtype=float)

        self.samples_ = reservoir
        self.fitted = self.fitted + X_.shape[0]

        # TODO
        # print(xp.sum(self.node_mass_[self.node_is_leaf_]))
        # print(self.fitted)
        return self

    def _grow(self, new_sample):
        xp, _ = get_array_module(new_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0]
        i_parent = l_leaf[self.search(new_sample, l_nodes=self.node_is_leaf_)[0, :]][0]
        lower_boundary_i = self.node_lower_boundaries[i_parent]
        upper_boundary_i = self.node_upper_boundaries[i_parent]
        self.samples_ = xp.concatenate([self.samples_, new_sample], axis=0)
        _, split_dim, split_value, _ = self._isolation_split(
            lower_boundary_i, upper_boundary_i
        )
        i_left, i_right = self.grow(i_parent, split_dim, split_value)
        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

        return i_parent, i_left, i_right

    def _prune(self, drop_sample):
        xp, _ = get_array_module(drop_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0]
        search_result = self.search(drop_sample, l_nodes=self.node_is_leaf_)
        if not xp.any(search_result):
            checked_and_warn(
                "[DEBUG] _prune is failing because the region of given sample is not found"
            )
        i_node_drop = l_leaf[search_result[0, :]][0]
        self.samples_ = xp.take(
            self.samples_,
            xp.where(xp.logical_not(xp.all(self.samples_ == drop_sample, axis=1)))[0],
            axis=0,
        )
        volume_drop = self.node_volumes_[i_node_drop]
        l_volumes_diff, l_keep = self.prune(i_node_drop)

        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()
        return i_node_drop, volume_drop, l_volumes_diff, l_keep

    def transform(self, X):
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, _ = rotate(X, self.SO_)
        else:
            X_ = X
        l_in = self.search(X_, self.node_is_leaf_)
        _, indices = xp.where(l_in)
        indices = xp.expand_dims(indices, axis=1)
        return indices


class AdaptiveMassEstimationTree(IsolationTree, DensityMixin):
    def __init__(self, psi, **kwargs):
        super().__init__(psi, **kwargs)
        self.fitted = 0

    def fit(self, X, y=None):
        xp, _ = get_array_module(X)
        super().fit(X, y)
        self.node_mass_ = xp.sum(self.search(X), axis=0, dtype=float)
        self.fitted = X.shape[0]
        # self.node_volumes_ = self.volumes() # included in fit

    def _grow(self, new_sample):
        xp, _ = get_array_module(new_sample)
        i_parent, i_left, i_right = super()._grow(new_sample)

        rho_parent = self.node_mass_[i_parent] / self.node_volumes_[i_parent]
        mass_left = rho_parent * self.node_volumes_[i_left]
        mass_right = rho_parent * self.node_volumes_[i_right]

        self.node_mass_ = xp.concatenate(
            [self.node_mass_, xp.array([mass_left, mass_right])]
        )
        return self

    def _prune(self, drop_sample):
        i_node_drop, volume_drop, l_volumes_diff, l_keep = super()._prune(drop_sample)
        rho_drop = self.node_mass_[i_node_drop] / volume_drop

        self.node_mass_ = self.node_mass_[l_keep]
        self.node_mass_ = self.node_mass_ + l_volumes_diff * rho_drop
        return self

    def score(self, X, y=None, return_demass=True):
        xp, _ = get_array_module(X)
        indices = super().transform(X)[:, 0]
        l_leaf_mass = self.node_mass_[self.node_is_leaf_]
        if return_demass:
            l_leaf_volumes = self.node_volumes_[self.node_is_leaf_]
            l_leaf_demass = l_leaf_mass / l_leaf_volumes
            return xp.take(l_leaf_demass, indices)
        else:
            return xp.take(l_leaf_mass, indices)
