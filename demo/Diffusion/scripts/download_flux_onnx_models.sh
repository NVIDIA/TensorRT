# Initialize Git LFS
git lfs install
export GIT_LFS_SKIP_SMUDGE=1

# Set argument defaults
arg_version="flux.1-dev"
arg_precision="bf16"
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --version) arg_version="$2"; shift;;
  --precision) arg_precision="$2"; shift;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help or -h                    : Print this help menu."
    echo " --version  <Flux Version>       : Choose one of ["flux.1-dev", "flux.1-schnell", "flux.1-dev-canny", "flux.1-dev-depth"]"
    echo " --precision  <Model Precision>  : Choose one of ["bf16", "fp8", "fp4"]"
    exit;
fi


if [ "$arg_version" = "flux.1-dev" ]; then
    # Clone the repository if it doesn't already exist
    onnx_dir="onnx-flux-dev"
    if [ ! -d "$onnx_dir" ] ; then
        git clone https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx $onnx_dir
    fi

    cd $onnx_dir
    git lfs pull --include=clip.opt
    git lfs pull --include=t5.opt
    git lfs pull --include=vae.opt
    cd transformer.opt
    if [ "$arg_precision" = "bf16" ]; then
        git lfs pull --include=bf16/
        mkdir -p ../transformer_bf16/transformer.opt
        ln -s $PWD/bf16/* $PWD/../transformer_bf16/transformer.opt
    elif [ "$arg_precision" = "fp8" ]; then
        git lfs pull --include=fp8/
        mkdir -p ../transformer_fp8/transformer-fp8.l4.0.bs2.s50.c32.p1.0.a0.8.opt/
        ln -s $PWD/fp8/* $PWD/../transformer_fp8/transformer-fp8.l4.0.bs2.s50.c32.p1.0.a0.8.opt/
    elif [ "$arg_precision" = "fp4" ]; then
        git lfs pull --include=fp4/
        mkdir -p ../transformer_fp4/transformer.opt
        ln -s $PWD/fp4/* $PWD/../transformer_fp4/transformer.opt
    else
        echo "Precision input $arg_precision not supported. Please choose one of ["bf16", "fp8", "fp4"]"
    cd ../..
    fi
elif [ "$arg_version" = "flux.1-schnell" ]; then
    # Clone the repository if it doesn't already exist
    onnx_dir="onnx-flux-schnell"
    if [ ! -d "$onnx_dir" ] ; then
        git clone https://huggingface.co/black-forest-labs/FLUX.1-schnell-onnx $onnx_dir
    fi

    cd $onnx_dir
    git lfs pull --include=clip.opt
    git lfs pull --include=t5.opt
    git lfs pull --include=vae.opt
    cd transformer.opt
    if [ "$arg_precision" = "bf16" ]; then
        git lfs pull --include=bf16/
        mkdir -p ../transformer_bf16/transformer.opt
        ln -s $PWD/bf16/* $PWD/../transformer_bf16/transformer.opt
    elif [ "$arg_precision" = "fp8" ]; then
        git lfs pull --include=fp8/
        mkdir -p ../transformer_fp8/transformer-fp8.l4.0.bs2.s50.c32.p1.0.a0.8.opt/
        ln -s $PWD/fp8/* $PWD/../transformer_fp8/transformer-fp8.l4.0.bs2.s50.c32.p1.0.a0.8.opt/
    elif [ "$arg_precision" = "fp4" ]; then
        git lfs pull --include=fp4/
        mkdir -p ../transformer_fp4/transformer.opt
        ln -s $PWD/fp4/* $PWD/../transformer_fp4/transformer.opt
    else
        echo "Precision input $arg_precision not supported. Please choose one of ["bf16", "fp8", "fp4"]"
    cd ../..
    fi
elif [ "$arg_version" = "flux.1-dev-depth" ]; then
    # Clone the repository if it doesn't already exist
    onnx_dir="onnx-flux-dev-depth"
    if [ ! -d "$onnx_dir" ] ; then
        git clone https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-onnx $onnx_dir
    fi

    cd $onnx_dir
    git lfs pull --include=clip.opt
    git lfs pull --include=t5.opt
    git lfs pull --include=vae.opt
    git lfs pull --include=vae_encoder.opt
    cd transformer.opt
    if [ "$arg_precision" = "bf16" ]; then
        git lfs pull --include=bf16/
        mkdir -p ../transformer_bf16/transformer.opt
        ln -s $PWD/bf16/* $PWD/../transformer_bf16/transformer.opt
    elif [ "$arg_precision" = "fp8" ]; then
        git lfs pull --include=fp8/
        mkdir -p ../transformer_fp8/transformer-fp8.l4.0.bs2.s30.c32.p1.0.a0.8.opt/
        ln -s $PWD/fp8/* $PWD/../transformer_fp8/transformer-fp8.l4.0.bs2.s30.c32.p1.0.a0.8.opt/
    elif [ "$arg_precision" = "fp4" ]; then
        git lfs pull --include=fp4/
        mkdir -p ../transformer_fp4/transformer.opt
        ln -s $PWD/fp4/* $PWD/../transformer_fp4/transformer.opt
    else
        echo "Precision input $arg_precision not supported. Please choose one of ["bf16", "fp8", "fp4"]"
    cd ../..
    fi
elif [ "$arg_version" = "flux.1-dev-canny" ]; then
    # Clone the repository if it doesn't already exist
    onnx_dir="onnx-flux-dev-canny"
    if [ ! -d "$onnx_dir" ] ; then
        git clone https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-onnx $onnx_dir
    fi

    cd $onnx_dir
    git lfs pull --include=clip.opt
    git lfs pull --include=t5.opt
    git lfs pull --include=vae.opt
    git lfs pull --include=vae_encoder.opt
    cd transformer.opt
    if [ "$arg_precision" = "bf16" ]; then
        git lfs pull --include=bf16/
        mkdir -p ../transformer_bf16/transformer.opt
        ln -s $PWD/bf16/* $PWD/../transformer_bf16/transformer.opt
    elif [ "$arg_precision" = "fp8" ]; then
        git lfs pull --include=fp8/
        mkdir -p ../transformer_fp8/transformer-fp8.l4.0.bs2.s30.c32.p1.0.a0.8.opt/
        ln -s $PWD/fp8/* $PWD/../transformer_fp8/transformer-fp8.l4.0.bs2.s30.c32.p1.0.a0.8.opt/
    elif [ "$arg_precision" = "fp4" ]; then
        git lfs pull --include=fp4/
        mkdir -p ../transformer_fp4/transformer.opt
        ln -s $PWD/fp4/* $PWD/../transformer_fp4/transformer.opt
    else
        echo "Precision input $arg_precision not supported. Please choose one of ["bf16", "fp8", "fp4"]"
    cd ../..
    fi
else
    echo "Version $arg_version not supported. Please choose one of ["flux.1-dev", "flux.1-schnell", "flux.1-dev-canny", "flux.1-dev-depth"]"
fi
