import hover_net
from hover_net.infer.tile import InferManager
from typing import Dict
import modal
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("hovernet-inference")
volume = modal.Volume.from_name("hovernet", create_if_missing=True)

# alternative: image = modal.Image.from_dockerfile("./Dockerfile", add_python="3.9")
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install([
        "libgl1-mesa-glx",
        "libopenslide0",
        "gcc",
        "python3-dev",
    ])
    .pip_install(
        "hover_net"
    )
)

@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume}, mounts=[modal.Mount.from_local_dir("./dataset", remote_path="/root/dataset")])
def run_hovernet_inference():
    logger.info("Starting HoVerNet inference")
    logger.info(f"hover_net version: {hover_net.__version__}")


    model_path = "/models/hovernet_fast_pannuke_type_tf2pytorch.tar"
    input_dir = "/root/dataset/tile/input"
    output_dir = "/root/dataset/tile/output"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input directory contents: {os.listdir(input_dir)}")
    logger.info(f"Looking for model at: {model_path}")
    logger.info(f"Current directory contents: {os.listdir('/')}")
    logger.info(f"Models directory contents: {os.listdir('/models')}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    method_args: Dict[str, Dict] = {
        'method': {
            'model_args': {
                'nr_types': 6,
                'mode': 'fast'
            },
            'model_path': model_path
        },
        'type_info_path': '/models/type_info.json'
    }

    logger.info(f"Method arguments: {method_args}")

    run_args = {
        'batch_size': 1,
        'nr_inference_workers': 8,
        'nr_post_proc_workers': 16,
        'patch_input_shape': 256,
        'patch_output_shape': 164,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'mem_usage': 0.2,
        'draw_dot': True,
        'save_qupath': True,
        'save_raw_map': True,
    }

    logger.info(f"Run arguments: {run_args}")

    logger.info("Initializing InferManager")
    infer = InferManager(**method_args)

    logger.debug(f"InferManager class: {infer.__class__}")
    logger.debug(f"InferManager attributes: {infer.__dict__}")
    
    logger.info("Starting file processing")

    try:
        infer.process_file_list(run_args)
        logger.info("File processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during file processing: {str(e)}", exc_info=True)
        raise

    logger.info("HoVerNet inference completed")

if __name__ == "__main__":
    logger.info("Starting Modal app")

    with modal.enable_output():
        with app.run(show_progress=False):
            run_hovernet_inference.remote()

    logger.info("Modal app execution completed")
