# OCSAug
OCSAug: Diffusion-based Optical Chemical Structure Data Augmentation for Improved Hand-drawn Chemical Structure Image Recognition

codebase

[MolScribe](https://github.com/thomas0809/MolScribe.git)

[RePaint](https://github.com/andreas128/RePaint.git)

[guided-diffusion](https://github.com/openai/guided-diffusion.git)

## Important Note Before You Begin
Before starting with the setup or execution of any scripts, please ensure to **check and modify the `file_path`** in any provided CSV files or scripts. The file paths must be adjusted to match your local system's directory structure. This is crucial for the correct functioning of the data processing, training, and evaluation scripts. Failing to do so may result in errors or incorrect processing of data.



## RePaint

### environment

```bash
cd RePaint
conda create -n repaint python
conda activate repaint
pip install -e .
pip install numpy torch blobfile tqdm pyYaml pillow # Example: torch 1.7.1+cu110.
conda install -c conda-forge mpi4py
conda install -c conda-forge openmpi
```

After setting up the environment, use `cd ..` to move back to the parent directory.

### DDPM train
Once back in the parent directory, change to the guided-diffusion directory:
```bash
cd guided-diffusion
mkdir ddpm_train_log
export OPENAI_LOGDIR=ddpm_train_log

MODEL_FLAGS=“—image_size 256 —attention_resolutions “32, 16, 8” —num_channerls 256 —num_head_channels 64 —num_res_blocks 2 —num_heads 4 —resblock_updown true —learn_sigma true —use_scale_shift_norm true —timestep_respacing “250”  —use_kl false —class_cond false —dropout 0.0“

DIFFUSION_FLAGS=“diffusion_steps 1000 —noise_schedule “linear” —rescale_learned sigmas false "

TRAIN_FLAGS=“—use_fp16 false  —batch_size 32 —microbatch 1 —lr 2e-5 —log_interval 100 —save_interval 10000”
```

```bash
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Note: `export OPENAI_LOGDIR` sets the location for DDPM training logs. If not set, logs are saved in the `/tmp` folder. The model checkpoints are saved at intervals specified by `save_interval` and are named in the format `opt_0.999_{step}`, `ema_0.999_{step}`, `model_0.999_{step}`. Use the `ema` prefixed file for sampling. Training does not have a predefined maximum step; it should be determined based on the quality of image samples from the checkpoints. The checkpoint for the model used in the paper is `ema_0.999_210000.pt`.

You can download this checkpoint from the following Google Drive link: [Download ema_0.999_210000.pt](https://drive.google.com/drive/folders/1VUrszbXm2FBVL6JzIH-0H5L1XSxMV7DL?usp=drive_link)


### RePaint

Configuration files are located in the confs folder.

Example configuration in molecule.yml:

```yaml
name: molecule_example
num_samples: 2,894
model_path: # Add your model file path here, e.g., './models/your_model_file.pt'
data:
  eval:
    paper_face_mask:
      mask_loader: true
      gt_path: # Path to the ground truth images, e.g., './data/datasets/gts/molecule'
      mask_path: # Path to the mask images, e.g., './data/datasets/gt_keep_masks/horizontal'
      image_size: 256
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 8
      paths: # Specify save paths for processed files
        srs: # Path to the sampling images, e.g., './log/molecule_example/inpainted'
        lrs: # Path where images with ground truth overlaid by masks are saved, e.g., './log/molecule_example/gt_masked'
        gts: # Path where ground truth images are saved, e.g., './log/molecule_example/gt'
        gt_keep_masks: # Path where mask images are saved, e.g., './log/molecule_example/gt_keep_mask'
```
After customizing `name`, `gt_path`, `mask_path`, and `paths` (`srs`, `lrs`, `gts`, `gt_keep_masks`):
Ensure that the number of files in `gt_path` matches the number of files in `mask_path` to maintain consistency during processing.

```bash
cd RePaint
python test.py --conf_path confs/molecule_example.yml
```

Sampled files are saved in ./log/molecule_example/inpainted.

### MolScribe

After completing the sampling, use `cd ..` from the RePaint directory to return to the parent directory.

### environment 


```bash
cd MolScribe

# Deactivate any active conda environments, such as RePaint, before proceeding.
conda deactivate

conda env create -f environment.yml
conda activate molscribe
mkdir -p ckpts
wget -P ckpts https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth
```

## Data Files
Access the implementation data through this [Google Drive link](https://drive.google.com/drive/folders/1VUrszbXm2FBVL6JzIH-0H5L1XSxMV7DL?usp=drive_link)
### Data
1. `filterered_DECIMER_train_train.csv` - Original training data
2. `filterered_DECIMER_train_val.csv` - Original validation data
3. `transform_rdkit.csv` - Original data augmented with RDKit images
4. `transform_randepict.csv` - Original data augmented with Randepict images
5. `transform_repaint.csv` - Original data augmented with RePaint images
6. `s_repaint_horizontal.csv` - Horizontal image and SMILES augmentation
7. `a_repaint_vertical.csv` - Vertical image and SMILES augmentation
8. `filterered_DECIMER_3194_test.csv` - Original test data

### Image Directories
1. `filterered_DECIMER_3194_train` - Contains original training and validation images.
2. `filterered_DECIMER_3194_test` - Contains images for testing.
3. `rdkit_image` - Contains images augmented using RDKit.
4. `Randepict_DECIMER_train_sets_OCSR` - Contains images augmented using Randepict.
5. `molecule_horizontal_example_1` - Contains images augmented using RePaint (horizontal).
6. `molecule_horizontal_example_2` - Contains images augmented using RePaint (horizontal).
7. `molecule_vertical_example_1` - Contains images augmented using RePaint (vertical).
8. `molecule_vertical_example_2` - Contains images augmented using RePaint (vertical).

### ZIP File Contents
When you access the Google Drive link, you will find the following ZIP files, each containing specific image folders:
- `DECIMER.zip` includes folders:
  1. `filterered_DECIMER_3194_train`
  2. `filterered_DECIMER_3194_test`
- `RDKit.zip` includes the folder:
  3. `rdkit_image`
- `Randepict.zip` includes the folder:
  4. `Randepict_DECIMER_train_sets_OCSR`
- `RePaint.zip` includes folders:
  5. `molecule_horizontal_example_1`
  6. `molecule_horizontal_example_2`
  7. `molecule_vertical_example_1`
  8. `molecule_vertical_example_2`
  

### Data Format for Training MolScribe

The CSV file used for training should include the following columns:

- `image_id`: Identifier for the image.
- `file_path`: File path corresponding to the `image_id`. 
- `SMILES`: SMILES notation associated with the `image_id`.

**Important:** You must modify the `file_path` column to reflect your local directory structure before using the CSV file. This ensures the paths are correctly set up for accessing the images.

## Train

```bash
bash train_scripts/train.sh
```
## train.sh example
```sh
#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0

BATCH_SIZE=64
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=train_output/
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    custom_train.py \
    --data_path data \
    --train_file DECIMER/DECIMER/transfrom_repaint.csv \
    --valid_file DECIMER/DECIMER/filtered_DECIMER_train_val.csv \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 256 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 2e-5 \
    --decoder_lr 2e-5 \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 30 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid \
    --fp16 --backend gloo \
    --load_path ckpts/swin_base_char_aux_1m680k.pth \
    --resume 2>&1
```

### Script Modification Options

- `NUM_GPUS_PER_NODE`: Number of GPUs per node
- `BATCH_SIZE`: Batch size
- `SAVE_PATH`: Path to save checkpoint files and prediction files for validation sets
- `train_file`: Path to training data
- `valid_file`: Path to validation data
- `epoch`: Number of epochs for fine-tuning

If you wish to train from scratch instead of fine-tuning, please refer to the MolScribe documentation.



## Predict

```bash
bash pred_scripts/predict.sh
```
## predict.sh example
```sh
#!/bin/bash

NUM_ITER=30 

BASE_DIR="train_decimer/transform_repaint"
IMAGE_PATH="data/DECIMER/DECIMER/filtered_DECIMER_3194_test"

for i in $(seq 1 $NUM_ITER); do
    MODEL_PATH="${BASE_DIR}/checkpoint_epoch_${i}.pth"
    SAVE_PATH="${BASE_DIR}/prediction_test_test_${i}.csv" 

    echo "save path: ${SAVE_PATH}" 

    torchrun custom_predict.py \
        --model_path ${MODEL_PATH} \
        --image_folder ${IMAGE_PATH} \
        --output_csv ${SAVE_PATH} \

done

echo "predict complete"

```
`single_predict.sh` - single checkpoint predict (executes prediction using a single model checkpoint.)

`predict.sh` - Iterative checkpoint predict (Repeatedly executes prediction using different model checkpoints across multiple iterations.)

## Evaluate

```bash
bash evaluate_scripts/single_eval.sh
```

```bash
bash evaluate_scripts/eval.sh
```

## eval.sh example
```sh
#!/bin/bash

NUM_ITER=30 

BASE_DIR="test_decimer/transform_repaint"
GOLD_FILE="data/DECIMER/DECIMER/filtered_DECIMER_test.csv"

echo "BASE_DIR : ${BASE_DIR}"
for i in $(seq 1 $NUM_ITER); do
    PRED_PATH="${BASE_DIR}/prediction_test_test_${i}.csv"

    torchrun evaluate.py \
        --gold_file ${GOLD_FILE} \
        --pred_file ${PRED_PATH} \
        --pred_field 'SMILES' \
        --tanimoto

done
echo "evaluation complete"

```
`single_eval.sh` - single evaluate (performs an evaluation using a single set of predictions.)
`eval.sh` - Iterative evaluate (Performs evaluations in a repeated manner across multiple sets of predictions.)
