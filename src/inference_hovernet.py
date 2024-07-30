# import modal
from hover_net.infer.tile import InferManager
import hover_net
from typing import Dict



if __name__ == "__main__":
    print("hover_net version: ", hover_net.__version__)


    method_args: Dict[str, Dict] = {
        'method': {
            'model_args': {
                'nr_types': 6,
                'mode': 'fast'
            },
            'model_path': 'pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar'
        },
        'type_info_path': 'type_info.json'
    }

    run_args = {
        'batch_size': 1,
        'nr_inference_workers': 8,
        'nr_post_proc_workers': 16,
        'patch_input_shape': 256,
        'patch_output_shape': 164,
        'input_dir': 'dataset/tile/input',
        'output_dir': 'dataset/tile/output',
        'mem_usage': 0.2,
        'draw_dot': True,
        'save_qupath': True,
        'save_raw_map': True,
    }

    infer = InferManager(**method_args)
    print(infer.__class__)
    print(infer.__dict__)
    
    infer.process_file_list(run_args)