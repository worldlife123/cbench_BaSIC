from cbench.benchmark.basic_benchmark import BenchmarkTestingWorker
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    BenchmarkTestingWorker,
).add_all_kwargs_as_param_slot()

