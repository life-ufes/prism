import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch

from pyro.infer import TraceEnum_ELBO, config_enumerate
from pyro.nn import PyroModule

from benchmarks.milk10k.bayesian.dataset import MILK10KBayesian
from benchmarks.milk10k.dataset import MILK10K


class Milk10kBayesianNetwork(PyroModule):
    """Bayesian network for MILK10K metadata."""

    FEATURE_NAMES = MILK10KBayesian.DEFAULT_FEATURES
    CATEGORICAL_FEATURES = MILK10KBayesian.CATEGORICAL_FEATURES
    CNN_FEATURES = MILK10KBayesian.CNN_PROB_FEATURES

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.num_classes = len(MILK10K.LABELS)
        self.eps = eps
        self.categorical_cardinality = {
            "sex": len(MILK10KBayesian.SEX_LEVELS),
            "skin_tone_class": len(MILK10KBayesian.SKIN_TONE_LEVELS),
            "site": len(MILK10KBayesian.SITE_LEVELS),
            "age_group": MILK10KBayesian.AGE_GROUP_CARDINALITY,
        }

    @config_enumerate
    def model(self, *feature_args, diagnosis_obs=None):
        if len(feature_args) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} feature tensors, received {len(feature_args)}."
            )

        observations = dict(zip(self.FEATURE_NAMES, feature_args))
        device = feature_args[0].device

        diagnosis_probs = pyro.param(
            "diagnosis_probs",
            torch.ones(self.num_classes, self.num_classes, device=device)
            / self.num_classes,
            constraint=constraints.simplex,
        )

        categorical_params = {
            name: pyro.param(
                f"{name}_probs",
                torch.ones(
                    self.num_classes, self.categorical_cardinality[name], device=device
                )
                / self.categorical_cardinality[name],
                constraint=constraints.simplex,
            )
            for name in self.CATEGORICAL_FEATURES
        }

        with pyro.plate("data", feature_args[0].shape[0]):
            cnn_probs = torch.stack(
                [observations[label] for label in self.CNN_FEATURES], dim=1
            )
            cnn = pyro.sample("cnn", dist.Categorical(probs=cnn_probs))
            diagnosis = pyro.sample(
                "diagnosis",
                dist.Categorical(probs=diagnosis_probs[cnn.long()]),
                obs=diagnosis_obs,
            )

            for name in self.CATEGORICAL_FEATURES:
                self._sample_categorical(
                    name, observations[name], categorical_params[name], diagnosis
                )

    def guide(self, *feature_args, diagnosis_obs=None):
        return

    def predict(self, *feature_args):
        marginals = TraceEnum_ELBO().compute_marginals(
            self.model, self.guide, *feature_args
        )
        probs = []
        device = feature_args[0].device
        for class_idx in range(self.num_classes):
            log_prob = marginals["diagnosis"].log_prob(
                torch.tensor(class_idx, device=device)
            )
            probs.append(log_prob.unsqueeze(0))
        probs = torch.exp(torch.cat(probs, dim=0)).T
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def _sample_categorical(self, name, obs, probs, diagnosis):
        valid = ~torch.isnan(obs)
        filled = obs.clone().long()
        filled[~valid] = 0
        distribution = dist.Categorical(probs=probs[diagnosis.long()]).mask(valid)
        pyro.sample(name, distribution, obs=filled)
