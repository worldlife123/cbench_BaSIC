from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.perceptual_output_layer import PerceptualOutputLayer

config = ClassBuilder(PerceptualOutputLayer,
    nn=ParamSlot(),
).add_all_kwargs_as_param_slot()