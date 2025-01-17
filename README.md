# OCSAug
OCSAug: Diffusion-based Optical Chemical Structure Data Augmentation for Improved Hand-drawn Chemical Structure Image Recognition

codebase

[RePaint](https://github.com/andreas128/RePaint.git)

[guided-diffusion](https://github.com/openai/guided-diffusion.git)


## OCSAug

### environment

```bash
conda create -n ocsaug python>=3.9
conda activate ocsaug
pip install numpy torch blobfile tqdm pyYaml pillow # Example: torch 1.7.1+cu110.
conda install -c conda-forge mpi4py
conda install -c conda-forge openmpi
pip install --upgrade gdown
```

## Data Download

Execute the following command to download the data:

```bash
bash download.sh
```

The downloaded data includes:

- `data/test_image_sample`: Sample test images
- `data/test_mask_sample`: Sample test masks
- `data/image`: Image data
- `data/csv`: CSV files
- `ckpt/molecule_model.pt`: Model checkpoint file

You can download the original data from [DECIMER - Hand-drawn molecule images dataset](https://zenodo.org/records/6456306). The provided data is derived from this original dataset.

## Train

Change to the guided-diffusion directory.
```bash
cd guided-diffusion
```

### Configuration Flags
```bash
mkdir ddpm_train_log
export OPENAI_LOGDIR=ddpm_train_log

MODEL_FLAGS="--image_size 256 --attention_resolutions '32, 16, 8' --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_heads 4 --resblock_updown true --learn_sigma true --use_scale_shift_norm true --timestep_respacing '250' --use_kl false --class_cond false --dropout 0.0"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule 'linear' --rescale_learned_sigmas false"

TRAIN_FLAGS="--use_fp16 false --batch_size 32 --lr 2e-5 --log_interval 100 --save_interval 10000"


```

`TRAIN_FLAGS` can be adjusted according to the user's needs. The parameters `use_fp16`, `batch_size`, `microbatch`, `lr`, `log_interval`, and `save_interval` should be set according to user preferences. The `batch_size` of 64 should be tailored based on the available VRAM. If you prefer a `batch_size` of 64 but have limited VRAM, you can adjust this setting using the `microbatch` option.

#### example `microbatch`
```bash
TRAIN_FLAGS=“—use_fp16 false  —batch_size 64 --microbatch 16—lr 2e-5 —log_interval 100 —save_interval 10000”
```

## Train scripts

```bash
python scripts/image_train.py --data_dir ../data/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Note: `export OPENAI_LOGDIR` sets the location for DDPM training logs. If not set, logs are saved in the `/tmp` folder. The model checkpoints are saved at intervals specified by `save_interval` and are named in the format `opt_0.999_{step}`, `ema_0.999_{step}`, `model_0.999_{step}`. Use the `ema` prefixed file for sampling. Training does not have a predefined maximum step; it should be determined based on the quality of image samples from the checkpoints. 

## Sampling

Change to the RePaint directory.
```bash
cd ../RePaint
```

```bash
python test.py --conf_path conf/molecule_example.yml
```
