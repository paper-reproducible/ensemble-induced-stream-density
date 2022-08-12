from sklearn.base import TransformerMixin, DensityMixin
from Common import (
    ReservoirSamplingEstimator,
    get_array_module,
    rotate,
    checked_and_warn,
)
from ._binary_tree import AxisParallelBinaryTree


def get_boundaries(X, ball_scaled=True):
    xp, _ = get_array_module(X)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    select_dims = xp.where(X_min < X_max)[0]

    if ball_scaled:
        global_upper_boundary = xp.ones([1, select_dims.shape[0]], dtype=X.dtype)
        global_lower_boundary = 0 - global_upper_boundary
    else:
        global_lower_boundary = xp.expand_dims(X_min[select_dims], axis=0)
        global_upper_boundary = xp.expand_dims(X_max[select_dims], axis=0)

    return global_lower_boundary, global_upper_boundary, select_dims


class IsolationTree(
    ReservoirSamplingEstimator, AxisParallelBinaryTree, TransformerMixin
):
    def __init__(self, psi, rotation=False, global_boundaries=None, **kwargs):
        super().__init__(psi)
        self.rotation = rotation
        self.global_boundaries = global_boundaries
        return

    def fit(self, X, y=None):
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, SO = rotate(X)
            self.SO_ = SO
        else:
            X_ = X
        if self.global_boundaries is not None:
            global_lower_boundary, global_upper_boundary = self.global_boundaries
            select_dims = xp.arange(X.shape[1])
        else:
            global_lower_boundary, global_upper_boundary, select_dims = get_boundaries(
                X_, self.rotation
            )
        self.select_dims = select_dims
        super().fit(X_[:, select_dims], y)
        super().seed(global_lower_boundary, global_upper_boundary)
        self._build_tree()
        return self

    def search(self, X, l_nodes=None):
        if self.rotation:
            X_, SO = rotate(X)
            self.SO_ = SO
        else:
            X_ = X
        X_ = X_[:, self.select_dims]
        return super().search(X_, l_nodes)

    def _isolation_split(self, i, stack_to_split, lower_boundary, upper_boundary):
        xp, _ = get_array_module(lower_boundary)
        eps = xp.finfo(self.samples_.dtype).eps

        l_in = self._search(self.samples_, lower_boundary, upper_boundary)
        l_in = l_in[:, 0]  # only one node therefore one column
        if xp.sum(l_in) == 1:
            return stack_to_split
        elif xp.sum(l_in) < 1:
            raise Exception("There is no sample in this region!")

        m_in = xp.take(self.samples_, xp.where(l_in)[0], axis=0)
        l_min = xp.min(m_in, axis=0)
        l_max = xp.max(m_in, axis=0)

        l_dims = xp.where(l_min + (2 * eps) <= l_max)[0]
        split_dim = xp.random.randint(low=0, high=l_dims.shape[0])
        split_dim = l_dims[split_dim]

        split_value = xp.random.uniform(l_min[split_dim], l_max[split_dim])
        if split_value == l_max[split_dim]:
            split_value = split_value - eps

        i_left, i_right = self.grow(i, split_dim, split_value)
        stack_to_split.append(i_left)
        stack_to_split.append(i_right)

        return stack_to_split

    def _build_tree(self):
        stack_to_split = [0]

        while len(stack_to_split) > 0:
            i = stack_to_split.pop()
            lower_boundary_i = self.node_lower_boundaries[i, :]
            upper_boundary_i = self.node_upper_boundaries[i, :]
            stack_to_split = self._isolation_split(
                i, stack_to_split, lower_boundary_i, upper_boundary_i
            )

        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

    def partial_fit(self, X, y=None):
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, _ = rotate(X, self.SO_)
        else:
            X_ = X
        X_ = X_[:, self.select_dims]
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
        l_in = self.search(X, self.node_is_leaf_)
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

    def score(self, X, y=None, return_demass=False):
        xp, _ = get_array_module(X)
        indices = super().transform(X)[:, 0]
        l_leaf_mass = self.node_mass_[self.node_is_leaf_]
        if return_demass:
            l_leaf_volumes = self.node_volumes_[self.node_is_leaf_]
            if xp.any(l_leaf_volumes == 0):
                eps = xp.finfo(X.dtype).eps
                l_leaf_demass = (l_leaf_mass + eps) / (l_leaf_volumes + eps)
            else:
                l_leaf_demass = l_leaf_mass / l_leaf_volumes
            return xp.take(l_leaf_demass, indices)
        else:
            return xp.take(l_leaf_mass, indices)
