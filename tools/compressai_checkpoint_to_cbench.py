import torch
from compressai.zoo.image import load_state_dict_from_url, model_urls
import os

def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def process(in_file, out_file, new_prefix, key_mapping, prefix_mapping, root="."):
    state_dict = torch.load(os.path.join(root, in_file))
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    new_state_dict = dict()
    for key, value in state_dict.items():
        should_discard = True
        new_key = new_prefix
        if key in key_mapping:
            target = key_mapping[key]
            if target is not None:
                should_discard = False
                new_key += key_mapping[key]
        else:
            for prefix in prefix_mapping:
                if key.startswith(prefix):
                    target = prefix_mapping[prefix]
                    if target is not None:
                        should_discard = False
                        new_key += target + key[len(prefix):]
                        break
        if not should_discard:
            print(f"Changing key {key} to {new_key}")
            new_state_dict[new_key] = value
        else:
            print(f"Discarding key {key}")

    torch.save(new_state_dict, os.path.join(root, out_file))

def mv_process_mbt2018(in_file, out_file_base, root="."):
    process(in_file, f"cbench-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "context_prediction.mask" : None,
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "context_prediction" : "latent_node_entropy_coders.y.context_prediction",
            "entropy_parameters" : "latent_node_entropy_coders.y.entropy_parameters",
            "g_a" : "latent_inference_modules.x_y.model",
            "h_a" : "latent_inference_modules.y_z.model",
            "g_s" : "latent_generative_modules.y_x.model",
            "h_s" : "latent_generative_modules.z_y.model",
        },
        root=root,
    )

    process(in_file, f"cbench-noy-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "context_prediction.mask" : None,
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "context_prediction" : "latent_node_entropy_coders.y.context_prediction",
            "entropy_parameters" : None, # "latent_node_entropy_coders.y.param_merger",
            "g_a" : "latent_inference_modules.x_y.model",
            "h_a" : "latent_inference_modules.y_z.model",
            "g_s" : "latent_generative_modules.y_x.model",
            "h_s" : "latent_generative_modules.z_y.model",
        },            
        root=root,
    )

    process(in_file, f"cbench-pgm-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "context_prediction.mask" : None,
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "context_prediction" : "latent_node_entropy_coders.y.context_prediction",
            "entropy_parameters" : "latent_node_entropy_coders.y.entropy_parameters",
            "g_a" : "latent_inference_modules.x_y.pgm_model",
            "h_a" : "latent_inference_modules.y_z.pgm_model",
            "g_s" : "latent_generative_modules.y_x.pgm_model",
            "h_s" : "latent_generative_modules.z_y.pgm_model",
        },
        root=root,
    )

    process(in_file, f"cbench-pgm-noy-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "context_prediction.mask" : None,
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "context_prediction" : "latent_node_entropy_coders.y.context_prediction",
            "entropy_parameters" : None, # "latent_node_entropy_coders.y.entropy_parameters",
            "g_a" : "latent_inference_modules.x_y.pgm_model",
            "h_a" : "latent_inference_modules.y_z.pgm_model",
            "g_s" : "latent_generative_modules.y_x.pgm_model",
            "h_s" : "latent_generative_modules.z_y.pgm_model",
        },
        root=root,
    )

def mv_process_bmshj2018_hyperprior(in_file, out_file_base, root="."):
    process(in_file, f"cbench-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "g_a" : "latent_inference_modules.x_y.model",
            "h_a" : "latent_inference_modules.y_z.model",
            "g_s" : "latent_generative_modules.y_x.model",
            "h_s" : "latent_generative_modules.z_y.model",
        },            
        root=root,
    )

    process(in_file, f"cbench-pgm-{out_file_base}",
        new_prefix="entropy_coder.",
        key_mapping={
            "entropy_bottleneck._offset" : None,
            "entropy_bottleneck._quantized_cdf" : None,
            "entropy_bottleneck._cdf_length" : None,
        },
        prefix_mapping={
            "entropy_bottleneck" : "latent_node_entropy_coders.z.entropy_bottleneck",
            "g_a" : "latent_inference_modules.x_y.pgm_model",
            "h_a" : "latent_inference_modules.y_z.pgm_model",
            "g_s" : "latent_generative_modules.y_x.pgm_model",
            "h_s" : "latent_generative_modules.z_y.pgm_model",
        },
        root=root,
    )


if __name__ == "__main__":
    root = "."
    
    for metric in ["mse", "ms-ssim"]:
        for quality in range(1, 5):
            model_url = model_urls["mbt2018"][metric][quality]
            load_state_dict_from_url(model_url, root)
            file_name = os.path.basename(model_url)
            file_name_simple = f"mbt2018-{metric}-{quality}.pth.tar"
            mv_process_mbt2018(file_name, file_name_simple, root=root)
            
    for metric in ["mse", "ms-ssim"]:
        for quality in range(1, 5):
            model_url = model_urls["bmshj2018-hyperprior"][metric][quality]
            load_state_dict_from_url(model_url, root)
            file_name = os.path.basename(model_url)
            file_name_simple = f"bmshj2018-hyperprior-{metric}-{quality}.pth.tar"
            mv_process_bmshj2018_hyperprior(file_name, file_name_simple, root=root)
