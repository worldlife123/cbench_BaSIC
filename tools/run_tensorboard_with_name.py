
import os

from configs.env import DEFAULT_EXPERIMENT_PATH

def main(args):
    tb_args = []
    log_dir = args.logdir
    for dir, subdirs, files in os.walk(log_dir):
        if os.path.samefile(dir, log_dir): continue # no need to replace root
        exp_name = None
        hash_string = os.path.basename(dir)[:8]
        for fname in files:
            if fname == args.config_filename:
                with open(os.path.join(dir, fname), 'r') as f:
                    exp_name = f.readline() # read the first line as name
        
        # TODO: there may exist multiple log files in dir
        # tensorboard keeps complaining...
        if exp_name is not None:
            tb_arg = f'"({hash_string}){exp_name}":"{dir}"'
            tb_args.append(tb_arg)

    run_cmd = "tensorboard --logdir_spec {}".format(",".join(tb_args))

    print("Running with: {}".format(run_cmd))

    os.system(run_cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", '-l', type=str, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--config-filename", '-c', type=str, default="exp_name.txt")
    # TODO: custom arguments to tensorboard...

    args = parser.parse_args()

    main(args)

