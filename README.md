# OCSAug
OCSAug: Diffusion-based Optical Chemical Structure Data Augmentation for Improved Hand-drawn Chemical Structure Image Recognition

codebase

[RePaint](https://github.com/andreas128/RePaint.git)

[guided-diffusion](https://github.com/openai/guided-diffusion.git)


## OCSAug

### Environment

Change to the guided-diffusion directory.
```bash
cd guided-diffusion
```

```bash
conda create -n ocsaug python
conda activate ocsaug
pip install -e .
pip install numpy torch blobfile tqdm pyYaml pillow # Example: torch 1.7.1+cu110.
conda install -c conda-forge mpi4py
conda install -c conda-forge openmpi
pip install --upgrade gdown
```

## Data Download

You can obtain all required files by running the following scripts:

```bash
# Download the pretrained diffusion model checkpoint
bash download.sh

# Download the OCSR dataset
bash download_for_OCSR.sh
```
Files downloaded by `bash download.sh`:

- `ckpt/molecule_model.pt`: Pretrained diffusion model checkpoint file

Files downloaded by `bash download_for_OCSR.sh`:
- `data/image`: Image data for OCSR
- `data/csv`: CSV files for OCSR
- `hand_drawn_image/hand_drawn_image` : Hand-drawn image sets

You can download the original data from [DECIMER - Hand-drawn molecule images dataset](https://zenodo.org/records/6456306). The provided data is derived from this original dataset.


`data`, `ckpt` : [Link1](https://drive.google.com/drive/folders/1VUrszbXm2FBVL6JzIH-0H5L1XSxMV7DL?usp=sharing)
`hand_drawn_image` : [Link2](https://drive.google.com/file/d/1Aetloltpf9FnXzYWt927RcQ7i5MOEdc5/view?usp=sharing)

## Sampling

Change to the RePaint directory.
```bash
cd ../RePaint
```

```bash
python test.py --conf_path conf/molecule_example.yml
```

If you wish to sample using a trained model, you will need to run the following script. Ensure that the `model_path` parameter in the `molecule_example.yml` configuration file is set to the correct path where your trained model is stored. This parameter is crucial for the script to locate and use the model correctly.

Following this section, the document will provide instructions on how to train your model, detailing the necessary steps and configurations needed to effectively train a model using the provided dataset and parameters.

## Train

Change to the guided-diffusion directory.
```bash
cd guided-diffusion
```

### Configuration Flags
```bash
mkdir ddpm_train_log
export OPENAI_LOGDIR=ddpm_train_log

MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 2 --num_heads 4 --num_head_channels 64 --attention_resolutions 32,16,8 --dropout 0.0 --use_checkpoint False --use_scale_shift_norm True --resblock_updown True --use_new_attention_order False --num_heads_upsample -1"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --use_kl False --predict_xstart False --rescale_timesteps False --rescale_learned_sigmas False"

TRAIN_FLAGS="--use_fp16 False --lr 2e-5 --batch_size 32 --log_interval 10 --save_interval 1000"


```

`TRAIN_FLAGS` can be adjusted according to the user's needs. The parameters `use_fp16`, `batch_size`, `microbatch`, `lr`, `log_interval`, and `save_interval` should be set according to user preferences. 

- **`log_interval`**: This parameter determines the frequency at which training progress logs are displayed in the terminal. Setting this interval helps in monitoring the training process more closely by providing updates on metrics like loss and accuracy at regular intervals.
  
- **`save_interval`**: This parameter specifies how often the model checkpoints are saved during training. Frequent saves can be useful for resuming training after interruptions or for evaluating the model at different stages of training.

The `batch_size` of 64 should be tailored based on the available VRAM. If you prefer a `batch_size` of 64 but have limited VRAM, you can adjust this setting using the `microbatch` option.


#### example `microbatch`
```bash
TRAIN_FLAGS="--use_fp16 False --lr 2e-5 --batch_size 32  --microbatch 1 --log_interval 10 --save_interval 1000"
```

## Train scripts

```bash
python scripts/image_train.py --data_dir ../data/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Note: `export OPENAI_LOGDIR` sets the location for DDPM training logs. If not set, logs are saved in the `/tmp` folder. The model checkpoints are saved at intervals specified by `save_interval` and are named in the format `opt_0.999_{step}`, `ema_0.999_{step}`, `model_0.999_{step}`. Use the `ema` prefixed file for sampling. Training does not have a predefined maximum step; it should be determined based on the quality of image samples from the checkpoints. 
