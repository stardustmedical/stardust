import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from hover_net import run_infer
from docopt import docopt

if __name__ == "__main__":
    # Set up the arguments
    argv = [
        "--gpu=0",
        "--nr_types=6",
        "--type_info_path=hover_net/type_info.json",
        "--batch_size=1",
        "--model_mode=fast",
        "--model_path=pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar",
        "--nr_inference_workers=8",
        "--nr_post_proc_workers=8",
        "tile",
        "--input_dir=dataset/sample_tiles/imgs/",
        "--output_dir=dataset/sample_tiles/pred/",
        "--mem_usage=0.1",
        "--draw_dot",
        "--save_qupath"
    ]

    # Parse arguments using docopt
    args = docopt(run_infer.__doc__, argv=argv, help=False, options_first=True, 
                  version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    # Parse sub-command arguments
    sub_cli_dict = {'tile': run_infer.tile_cli, 'wsi': run_infer.wsi_cli}
    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)

    # Set up CUDA
    import torch
    import os
    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    nr_gpus = torch.cuda.device_count()
    run_infer.log_info('Detect #GPUS: %d' % nr_gpus)

    # Process arguments
    args = {k.replace('--', '') : v for k, v in args.items()}
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}

    # Set up method arguments
    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : args['model_mode'],
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' \
                            else args['type_info_path'],
    }

    # Set up run arguments
    run_args = {
        'batch_size' : max(1, int(args['batch_size']) * nr_gpus),  # Ensure minimum batch size of 1
        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],
            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : sub_args['draw_dot'],
            'save_qupath' : sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
        })
        from hover_net.infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    elif sub_cmd == 'wsi':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],
            'input_mask_dir' : sub_args['input_mask_dir'],
            'cache_path'     : sub_args['cache_path'],
            'proc_mag'       : int(sub_args['proc_mag']),
            'ambiguous_size' : int(sub_args['ambiguous_size']),
            'chunk_shape'    : int(sub_args['chunk_shape']),
            'tile_shape'     : int(sub_args['tile_shape']),
            'save_thumb'     : sub_args['save_thumb'],
            'save_mask'      : sub_args['save_mask'],
        })
        from hover_net.infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)