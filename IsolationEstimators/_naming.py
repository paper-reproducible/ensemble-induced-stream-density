from enum import Enum
from ._voronoi import VoronoiPartitioning
from ._voronoi_soft import SoftVoronoiPartitioning
from ._inn import INNPartitioning
from ._isolation_tree import IsolationTree
from ._fuzzy import FuzzyPartitioning


class PartitioningMethod(Enum):
    VORONOI = "voronoi"
    SOFT_VORONOI = "soft_voronoi"
    INN = "inn"
    ITREE = "itree"
    GAUSSIAN = "gaussian"


class IsolationModel(Enum):
    ANNE = "anne"
    INNE = "inne"
    SOFT_ANNE = "soft_anne"
    IFOREST = "iforest"
    FUZZY = "fuzzy"


class EstimatorType(Enum):
    MASS = "mass"
    ANOMALY = "anomaly"
    KERNEL = "kernel"
    DENSITY = "density"


model_partitioning_map = {
    IsolationModel.ANNE: PartitioningMethod.VORONOI,
    IsolationModel.INNE: PartitioningMethod.INN,
    IsolationModel.SOFT_ANNE: PartitioningMethod.SOFT_VORONOI,
    IsolationModel.IFOREST: PartitioningMethod.ITREE,
    IsolationModel.FUZZY: PartitioningMethod.GAUSSIAN,
}


def get_partitioning_initializer(isolation_model, psi, **kwargs):
    partitioning_method = model_partitioning_map[
        isolation_model
        if isinstance(isolation_model, IsolationModel)
        else IsolationModel(isolation_model)
    ]
    if partitioning_method == PartitioningMethod.VORONOI:
        return lambda: VoronoiPartitioning(psi, **kwargs)
    if partitioning_method == PartitioningMethod.SOFT_VORONOI:
        return lambda: SoftVoronoiPartitioning(psi, **kwargs)
    if partitioning_method == PartitioningMethod.INN:
        return lambda: INNPartitioning(psi, **kwargs)
    if partitioning_method == PartitioningMethod.ITREE:
        return lambda: IsolationTree(psi, **kwargs)
    if partitioning_method == PartitioningMethod.GAUSSIAN:
        return lambda: FuzzyPartitioning(psi, random_scale=True, **kwargs)
    raise NotImplementedError()
