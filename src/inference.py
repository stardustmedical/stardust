import sys
import os
from typing import Literal
import logging
import torch
import modal
import torch
import os
import traceback
import time

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "opencv-python-headless",
        "docopt",
        "numpy",
        # "git+https://github.com/leandermaerkisch/hover_net.git@fbc8a3906e3db61a1844b400b806c3dbf5e6b55f#egg=hover_net"
    )
)

app = modal.App("hover-net-inference", image=image)

@app.function(image=image, gpu="A10G", timeout=3600)
def run_inference(data_type):
    import sys
    import os
    import subprocess
    import site


    import traceback
    from typing import Literal
    import logging
    import torch
    import time
    
    print("Starting run_inference function")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")

    print(f"Python path: {sys.path}")

    # Print site-packages directories
    print("Site-packages directories:")
    for path in site.getsitepackages():
        print(f"  {path}")
        if os.path.exists(path):
            print(f"    Contents: {os.listdir(path)}")
    
    # Print installed packages
    print("Installed packages:")
    subprocess.run([sys.executable, "-m", "pip", "list"])

    with image.imports():
        # Try to install hover_net
        print("Attempting to install hover_net...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/leandermaerkisch/hover_net.git@2759d24e34df3db38550c646726dab4ca23f5edf#egg=hover_net"])
    
        # Function to search for files
        def find_files(name, path):
            result = []
            for root, dirs, files in os.walk(path):
                if name in files or name in dirs:
                    result.append(os.path.join(root, name))
            return result

        # Search for hover_net in common locations
        search_paths = ['/usr/local/lib/python3.*/site-packages', '/root/.local/lib/python3.*/site-packages', '/opt/conda/lib/python3.*/site-packages']
        
        for search_path in search_paths:
            hover_net_locations = find_files('hover_net', search_path)
            if hover_net_locations:
                print(f"Found hover_net in the following locations:")
                for location in hover_net_locations:
                    print(f"  {location}")
                    if os.path.isdir(location):
                        print(f"  Contents: {os.listdir(location)}")
            else:
                print(f"hover_net not found in {search_path}")

        # Try to import hover_net
        try:
            import hover_net
            print(f"hover_net package location: {hover_net.__file__}")
            from hover_net import run_infer
            print("Successfully imported run_infer from hover_net")
        except ImportError as e:
            print(f"Error importing hover_net: {e}")
            print("Traceback:")
            traceback.print_exc()

        # If import fails, try to find hover_net manually
        if 'hover_net' not in sys.modules:
            print("Manual search for hover_net:")
            for path in sys.path:
                hover_net_path = os.path.join(path, 'hover_net')
                if os.path.exists(hover_net_path):
                    print(f"Found hover_net directory at: {hover_net_path}")
                    print(f"Contents: {os.listdir(hover_net_path)}")

        return  # Exit the function if we can't import hover_net

    from docopt import docopt


    def log_with_timestamp(message):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

        # Set up the arguments
        data_type = (
            sys.argv[1] if len(sys.argv) > 1 else "tile"
        )  # Default to 'tile' if no argument is provided
        print(f"Running in data type processing mode: {data_type}")

    log_with_timestamp("Setting up arguments")

    common_args = [
        "--gpu=0",
        "--nr_types=6",
        "--type_info_path=hover_net/type_info.json",
        "--batch_size=1",
        "--model_mode=fast",
        "--model_path=pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar",
        "--nr_inference_workers=8",
        "--nr_post_proc_workers=8",
    ]

    data_type_args = {
        "tile": [
            "--input_dir=dataset/sample_tiles/imgs/",
            "--output_dir=dataset/sample_tiles/pred/",
            "--mem_usage=0.1",
            "--draw_dot",
            "--save_qupath",
        ],
        "wsi": [
            "--input_dir=dataset/sample_wsi/imgs",
            "--output_dir=dataset/sample_tiles/pred/",
            "--proc_mag=40",
            "--cache_path=./cache/",
            "--chunk_shape=1024",
            "--tile_shape=256",
            "--save_thumb",
            "--save_mask",
        ],
    }

    log_with_timestamp("Parsing arguments with docopt")

    argv = common_args + [data_type] + data_type_args[data_type]

    try:
        # Parse arguments using docopt
        args = docopt(
            run_infer.__doc__,
            argv=argv,
            help=False,
            options_first=True,
            version="HoVer-Net Pytorch Inference v1.0",
        )
        # ... rest of the code ...
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return  # Exit the function if an error occurs

    sub_cmd = args.pop("<command>")
    sub_cmd_args = args.pop("<args>")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

    # Parse sub-command arguments
    sub_cli_dict = {"tile": run_infer.tile_cli, "wsi": run_infer.wsi_cli}
    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)

    log_with_timestamp("Setting up CUDA")
    args.pop("--version")
    gpu_list = args.pop("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    nr_gpus = torch.cuda.device_count()
    run_infer.log_info("Detect #GPUS: %d" % nr_gpus)

    log_with_timestamp("Processing arguments")
    args = {k.replace("--", ""): v for k, v in args.items()}
    sub_args = {k.replace("--", ""): v for k, v in sub_args.items()}

    log_with_timestamp("Setting up method arguments")
    nr_types = int(args["nr_types"]) if int(args["nr_types"]) > 0 else None
    method_args = {
        "method": {
            "model_args": {
                "nr_types": nr_types,
                "mode": args["model_mode"],
            },
            "model_path": args["model_path"],
        },
        "type_info_path": None
        if args["type_info_path"] == ""
        else args["type_info_path"],
    }

    # Set up run arguments
    run_args = {
        "batch_size": max(1, int(args["batch_size"]) * nr_gpus),
        "nr_inference_workers": int(args["nr_inference_workers"]),
        "nr_post_proc_workers": int(args["nr_post_proc_workers"]),
        "patch_input_shape": 256 if args["model_mode"] == "fast" else 270,
        "patch_output_shape": 164 if args["model_mode"] == "fast" else 80,
    }

    # Update run_args with data type-specific arguments
    for k, v in sub_args.items():
        key = k.replace("--", "")
        if key == "mem_usage":
            run_args[key] = float(v)  # Convert mem_usage to float
        elif key in ["proc_mag", "chunk_shape", "tile_shape"]:
            run_args[key] = int(v)  # Convert these to integers
        else:
            run_args[key] = v

    log_with_timestamp(f"Importing InferManager for {sub_cmd}")
    if sub_cmd == "tile":
        from hover_net.infer.tile import InferManager

        log_with_timestamp("Starting tile processing...")
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    elif sub_cmd == "wsi":
        log_with_timestamp("Starting WSI processing")
        from hover_net.infer.wsi import InferManager

        infer = InferManager(**method_args)
        try:
            log_with_timestamp(f"Input directory: {run_args['input_dir']}")
            log_with_timestamp(f"Output directory: {run_args['output_dir']}")

            # List files in the input directory
            wsi_files = []

            log_with_timestamp("Starting WSI processing...")

            for root, dirs, files in os.walk(run_args["input_dir"]):
                for file in files:
                    if file.endswith(
                        ".ndpi"
                    ):  # or whatever extension your WSI files have
                        wsi_files.append(os.path.join(root, file))

            log_with_timestamp(f"Found {len(wsi_files)} WSI files")

            for idx, file in enumerate(wsi_files):
                log_with_timestamp(f"Processing file {idx+1}/{len(wsi_files)}: {file}")
                start_time = time.time()

                # Process a single WSI file
                wsi_path = file
                msk_path = None  # We don't have a mask path, so we'll pass None
                output_dir = os.path.join(
                    run_args["output_dir"], os.path.basename(file).split(".")[0]
                )
                os.makedirs(output_dir, exist_ok=True)

                infer.process_single_file(wsi_path, msk_path, output_dir)

                end_time = time.time()
                log_with_timestamp(
                    f"Finished processing {file}. Time taken: {end_time - start_time:.2f} seconds"
                )

            log_with_timestamp("WSI processing completed.")
        except Exception as e:
            print(f"An error occurred during WSI processing: {e}")
            traceback.print_exc()

    log_with_timestamp("Finished run_inference function")


if __name__ == "__main__":
    with app.run():
        data_type: Literal["wsi", "tile"] = "tile"
        print("Starting remote execution of run_inference")
        run_inference.remote(data_type)
        print("Finished remote execution of run_inference")
