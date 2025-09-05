#!/usr/bin/env python3

# Original converter: https://huggingface.co/spaces/tumuyan2/model2mnn

import argparse
parser = argparse.ArgumentParser(description='PyTorch to ONNX models converter.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', '-i', default=None,
                    help='PyTorch or ONNX model')
parser.add_argument('--onnx-model', '-o', default=None,
                    help='intermediate ONNX model')
parser.add_argument('--ncnn-model', '-n', default=None,
                    help='NCNN model, use "-" to skip')
parser.add_argument('--mnn-model', '-m', default=None,
                    help='MNN model, use "-" to skip')
parser.add_argument('--shape-size', '-s', type=int, default=16,
                    help='shape size')
parser.add_argument('--opset-version', '-v', type=int, default=13,
                    help='ONNX opset version')
parser.add_argument('--fp16', '-f', default=False, action='store_true',
                    help='export FP16 models'),
parser.add_argument('--simplify', '-l', default=False, action='store_true',
                    help='simplify models'),
args = parser.parse_args()
import os, re, time
def print_t(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)
def print_nt(*args, **kwargs):
    print(time.strftime("\n[%H:%M:%S]"), *args, **kwargs)

starttime = time.time()
print_t("Running...")

input_path = args.model
onnx_path  = args.onnx_model
ncnn_path  = args.ncnn_model
mnn_path   = args.mnn_model
tilesize   = args.shape_size
use_fp16   = args.fp16
sim_opt    = args.simplify
opset      = args.opset_version
channel    = 0
scale      = 0

if input_path is None:
    input_files = [f for f in os.listdir(".") if f.lower().endswith(".pth")]
    if not input_files:
        input_files = [f for f in os.listdir(".") if f.lower().endswith(".pt")]
        if not input_files:
            input_files = [f for f in os.listdir(".") if f.lower().endswith(".onnx")]
            if not input_files:
                print_nt("Neither PyTorch or ONNX model is specified or found in current directory.\nStop.\n")
                quit(-1)
    input_path = input_files[0]
    print_nt(f"Model is not specified, found \"./{input_path}\".\n")
else:
    print_nt(f"Model: {input_path}\n")

print_t(f"Using {"half" if use_fp16 else "single"} floats precision.")
import onnx
print_t("Loaded ONNX.")

if not input_path.endswith('.onnx'):
    import torch
    print_t("Loaded Torch.")
    from spandrel import ImageModelDescriptor, ModelLoader
    print_t("Loaded Spandrel.")

    model_descriptor = ModelLoader().load_from_file(input_path)

    # Ensure it's the expected type from Spandrel
    if not isinstance(model_descriptor, ImageModelDescriptor):
        print_t(f"Error: Expected ImageModelDescriptor, but got {type(model_descriptor)}")
        print_t("Please ensure the .pth file is compatible with Spandrel's loading mechanism.")

    # Get the underlying torch.nn.Module
    torch_model = model_descriptor.model

    # Set the model to evaluation mode (important for dropout, batchnorm layers)
    torch_model.eval()

    channel = model_descriptor.input_channels
    example_input = torch.randn(1, channel, tilesize, tilesize, requires_grad=False)
    print_t(f"Model input channels: {channel}, using shape 1x{channel}x{tilesize}x{tilesize}.")

    '''
    # export to FP16 is too slow
    if use_fp16:
        torch_model.half()
        example_input = example_input.half()
        print_t("Using half precision.")
    #'''
    print_t(f"Model loaded successfully: {type(torch_model).__name__}.")

    if onnx_path is None:
        base_path = os.path.splitext(input_path)[0]

        scale = model_descriptor.scale
        # Determine whether the file name of pth_path contains xN or Nx
        filename = os.path.basename(input_path).upper()
        pattern = f'(^|[_-])({scale}X|X{scale})([_-]|$)'
        if re.search(pattern, filename):
            print_t(f'File name contains scale info: {filename}, ignoring scale {scale}.')
        else:
            print_t(f'Model scale: {scale}.')
            base_path = f"{base_path}-x{scale}"
        onnx_path = base_path + ("-Grayscale" if channel==1 else "") + ".onnx"
        pattern = f'(^|[_-])(X{scale})([_-]|$)'
        if re.search(pattern, filename):
            scale = 0

    print_t(f"ONNX path: '{onnx_path}'.")
    print_t(f"ONNX model exporting (opset version {opset})...")
    axes = {}

    torch.onnx.export(
        torch_model,                 # The model instance
        example_input,               # An example input tensor
        onnx_path,                   # Where to save the model (file path)
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=opset,         # The ONNX version to export the model to (choose based on target runtime)
        do_constant_folding=True,    # Whether to execute constant folding for optimization
        input_names=['input'],       # The model's input names
        output_names=['output'],     # The model's output names
        autograd_inlining=False,
        dynamic_axes=axes
    )

    print_nt(f"ONNX model exported successfully: {onnx_path}.\n")
    torch_model = None
else:
    onnx_path = input_path
    tilesize = 0

if sim_opt:
    from onnxsim import simplify
    print_t("Loaded ONNX simplifier.")
    print_t("Simplifying ONNX model...")
    sim_path = os.path.splitext(onnx_path)[0] + "_sim.onnx"
    onnx_model = onnx.load(onnx_path)
    model_sim = simplify(onnx_model, perform_optimization=True)[0]
    onnx.save(model_sim, sim_path)
    print_nt(f"Simplified ONNX model exported successfully: {sim_path}.\n")
    onnx_path = sim_path

import subprocess
if "-" != ncnn_path:
    print_t("Exporting NCNN model...")
    if ncnn_path is None:
        ncnn_path = os.path.splitext(onnx_path)[0]
    elif ncnn_path.endswith('.bin') or ncnn_path.endswith('.param'):
        ncnn_path = os.path.splitext(ncnn_path)[0]
    if scale:
        ncnn_path += f"-x{scale}"
    try:
        command = f"onnx2ncnn \"{onnx_path}\" \"{ncnn_path}.param\" \"{ncnn_path}.bin\""
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '':
                if process.poll() is not None:
                    break
            else:
                print(output.strip())
            time.sleep(0.1)
        returncode = process.poll()
        if returncode != 0:
            print_nt(f"onnx2ncnn returned {returncode}, command was:\n{command}")
            print_t("(-11 probably means segmentation fault in onnx2ncnn).\n" if returncode == -11 else "")
    except Exception as e:
        print_nt(f"onnx2ncnn process returned exception: {str(e)}\n")

    ncnn_path, ncnn_name = os.path.split(ncnn_path)
    if ncnn_path is None or len(ncnn_path) == 0:
        ncnn_path = '.'
    param_files = [f for f in os.listdir(ncnn_path) if f.endswith(f"{ncnn_name}.param")]
    bin_files   = [f for f in os.listdir(ncnn_path) if f.endswith(f"{ncnn_name}.bin")]
    if param_files and bin_files:
        print_nt(f"NCNN model exported successfully: \"{ncnn_path}/{ncnn_name}.param\" + \"{ncnn_name}.bin\".\n")

    if sim_opt or use_fp16:
        print_t("Optimising NCNN model...")
        opt_name = ncnn_name + ("_opt-fp16" if use_fp16 else "_opt")
        try:
            command = "ncnnoptimize " + \
                f"\"{ncnn_path}/{ncnn_name}.param\" \"{ncnn_path}/{ncnn_name}.bin\" " + \
                f"\"{ncnn_path}/{opt_name}.param\"  \"{ncnn_path}/{opt_name}.bin\"  " + \
                f"{1 if use_fp16 else 0}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                output = process.stdout.readline()
                if output == '':
                    if process.poll() is not None:
                        break
                else:
                    print(output.strip())
                time.sleep(0.1)
            returncode = process.poll()
            if returncode != 0:
                print_nt(f"ncnnoptimize returned {returncode}, command was:\n{command}")
                print_t("(-11 probably means segmentation fault in ncnnoptimize).\n" if returncode == -11 else "")
        except Exception as e:
            print_nt(f"ncnnoptimize process returned exception: {str(e)}\n")
        param_files = [f for f in os.listdir(ncnn_path) if f.endswith(f"{opt_name}.param")]
        bin_files   = [f for f in os.listdir(ncnn_path) if f.endswith(f"{opt_name}.bin")]
        if param_files and bin_files:
            print_nt(f"NCNN model optimized successfully: \"{ncnn_path}/{opt_name}.param\" + \"{opt_name}.bin\".\n")

if "-" != mnn_path:
    print_t("Exporting MNN model...")
    if mnn_path is None:
        mnn_path = os.path.splitext(onnx_path)[0] + ".mnn"
    else:
        mnn_path += ("" if not mnn_path.lower().endswith(".mnn") else ".mnn")

    try:
        command = f"MNNConvert -f ONNX --modelFile \"{onnx_path}\" --MNNModel \"{mnn_path}\"" + (" --fp16" if use_fp16 else "")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '':
                if process.poll() is not None:
                    break
            else:
                print(output.strip())
            time.sleep(0.1)
        returncode = process.poll()
        if returncode != 0:
            print_nt(f"MNNConvert returned {returncode}, command was:\n{command}")
            print_t("(-11 probably means segmentation fault in MNNConvert).\n" if returncode == -11 else "")
    except Exception as e:
        print_nt(f"MNNConvert process returned exception: {str(e)}\n")

    if os.path.exists(mnn_path):
        print_nt(f"MNN model exported successfully: {mnn_path}\n")

        print_t("Resetting MNN input shape and model ID...")
        f = open(mnn_path, "r+b")
        f.seek(40)
        m = f.read(4)
        o = m[0] + (m[1] << 8) + (m[2] << 16) + (m[3] << 24)
        f.seek(o - 8) # shape offset
        m = f.read(4)
        s1 = m[0] + (m[1] << 8) + (m[2] << 16) + (m[3] << 24)
        m = f.read(4)
        s2 = m[0] + (m[1] << 8) + (m[2] << 16) + (m[3] << 24)
        m = f.read(4)
        s3 = m[0] + (m[1] << 8) + (m[2] << 16) + (m[3] << 24)
        m = f.read(4)
        s4 = m[0] + (m[1] << 8) + (m[2] << 16) + (m[3] << 24)
        if channel and tilesize and (s1 != 1 or s2 != channel or s3 != tilesize or s4 != tilesize):
            print_t(f"Warning: model input shape {s1}x{s2}x{s3}x{s4} does not match desired {s1}x{channel}x{tilesize}x{tilesize}.")
        f.seek(o)
        m = "\xff\xff\xff\xff".encode('latin-1')
        f.write(m)
        f.write(m)
        #print_t(f"Input shape: {s1}x{s2}x{s3}x{s4} -> {s1}x{s2}x(-1)x(-1)")
        m = "00000000-0000-0000-0000-000000000000".encode()
        f.seek(72)
        f.write(m)
        f.close()
        m = ""
        try:
            command = f"md5sum \"{mnn_path}\""
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                output = process.stdout.readline()
                if output == '':
                    if process.poll() is not None:
                        break
                else:
                    m += output.strip()
                time.sleep(0.1)
            returncode = process.poll()
            if returncode != 0:
                print_nt(f"md5sum returned {returncode}, command was:\n{command}")
                print_t("(-11 probably means segmentation fault in md5sum).\n" if returncode == -11 else "")
        except Exception as e:
            print_nt(f"md5sum process returned exception: {str(e)}\n")
            quit()
        m = f"{m[0:8]}-{m[8:12]}-{m[12:16]}-{m[16:20]}-{m[20:32]}".encode()
        f = open(mnn_path, "r+b")
        f.seek(72)
        f.write(m)
        f.close()
starttime = time.time() - starttime
print_t(f"Done (taken {int(starttime // 60)} minutes {int(starttime % 60 // 1)} seconds).")
