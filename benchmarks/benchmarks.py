from enum import Enum


class Benchmarks(Enum):
    PAD20 = "pad20"
    MILK10K = "milk10k"


class BenchmarksFactory:
    @staticmethod
    def get_dataset(bench: Benchmarks):
        if bench == Benchmarks.PAD20:
            from benchmarks.pad20.dataset import PAD20

            return PAD20
        elif bench == Benchmarks.MILK10K:
            from benchmarks.milk10k.dataset import MILK10K

            return MILK10K
        else:
            raise ValueError(f"The dataset {bench} is not available.")

    @staticmethod
    def get_experiment(benchmark):
        if benchmark == Benchmarks.PAD20:
            from benchmarks.pad20.experiment import ex as experiment_pad_20

            return experiment_pad_20
        elif benchmark == Benchmarks.MILK10K:
            from benchmarks.milk10k.experiment import ex as experiment_milk10k

            return experiment_milk10k
        else:
            raise ValueError(f"The benchmark {benchmark} is not available.")

    @staticmethod
    def get_bayesian_experiment(benchmark):
        if benchmark == Benchmarks.PAD20:
            from benchmarks.pad20.bayesian.train import ex as ex_bayesian_pad_20

            return ex_bayesian_pad_20
        elif benchmark == Benchmarks.MILK10K:
            from benchmarks.milk10k.bayesian.train import ex as ex_bayesian_milk10k

            return ex_bayesian_milk10k
        else:
            raise ValueError(f"The benchmark {benchmark} is not available.")
