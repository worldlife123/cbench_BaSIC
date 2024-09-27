from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.binary_codec import BPG
from configs.env import BPG_ENCODER_PATH, BPG_DECODER_PATH

config = ClassBuilder(
    BPG,
    "--encoder-path",
    BPG_ENCODER_PATH,
    "--decoder-path",
    BPG_DECODER_PATH,
    # encoder_path=BPG_ENCODER_PATH,
    # decoder_path=BPG_DECODER_PATH,
).add_all_kwargs_as_param_slot()