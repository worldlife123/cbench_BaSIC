from configs.class_builder import ClassBuilder, ParamSlot
import cbench.codecs

config = ClassBuilder(
    cbench.codecs.ZstdDictWrapperCodec,
).add_all_kwargs_as_param_slot().update_args(
    dict_size=ParamSlot("dict_size", default=32*1024) # 32KB
    # compressor_config=ParamSlot("level", 
    #     choices={i: dict(level=i) for i in range(1, 23)},
    #     default=dict(level=3),
    # )
)