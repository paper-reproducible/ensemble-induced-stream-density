from ._naming import (
    IsolationModel,
    model_partitioning_map,
    EstimatorType,
    PartitioningMethod,
)
from ._data_dependent import (
    DEMassEstimator,
    MassEstimator,
    IsolationTransformer,
    IncrementalMassEstimator,
)
from ._data_independent import DataIndependentEstimator, DataIndependentDensityEstimator
from ._outlier_iforest import IsolationForestAnomalyDetector
from ._outlier_mass import IsolationBasedAnomalyDetector


def init_estimator(
    estimator_type, isolation_model, psi, t=100, data_dependent=True, **kwargs
):
    partitioning_method = model_partitioning_map[
        isolation_model
        if isinstance(isolation_model, IsolationModel)
        else IsolationModel(isolation_model)
    ].value

    if estimator_type in [EstimatorType.DENSITY, EstimatorType.DENSITY.value]:
        if data_dependent:
            if partitioning_method == PartitioningMethod.ITREE:
                return DEMassEstimator(psi, t, **kwargs)
            else:
                raise NotImplementedError()
        else:
            if partitioning_method == PartitioningMethod.ITREE:
                return DataIndependentDensityEstimator(
                    psi, t, isolation_model=isolation_model, **kwargs
                )
            else:
                return DataIndependentEstimator(psi, t, isolation_model, **kwargs)

    if estimator_type in [EstimatorType.MASS, EstimatorType.MASS.value]:
        if data_dependent:
            if partitioning_method == PartitioningMethod.ITREE:
                return IncrementalMassEstimator(psi, t, **kwargs)
            else:
                return MassEstimator(psi, t, isolation_model=isolation_model, **kwargs)
        else:
            return DataIndependentEstimator(psi, t, isolation_model=isolation_model)

    if estimator_type in [EstimatorType.ANOMALY, EstimatorType.ANOMALY.value]:
        if data_dependent:
            if isolation_model in [
                IsolationModel.IFOREST,
                IsolationModel.IFOREST.value,
            ]:
                return IsolationForestAnomalyDetector(psi, t, **kwargs)
            else:
                return IsolationBasedAnomalyDetector(
                    psi, t, isolation_model=isolation_model, **kwargs
                )
        else:
            raise NotImplementedError()

    if estimator_type in [EstimatorType.KERNEL, EstimatorType.KERNEL.value]:
        if data_dependent:
            return IsolationTransformer(
                psi, t, isolation_model=isolation_model, **kwargs
            )
        else:
            raise NotImplementedError()

    raise NotImplementedError()
