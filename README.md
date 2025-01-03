# OCSAug
OCSAug: Diffusion-based Optical Chemical Structure Data Augmentation for Improved Hand-drawn Chemical Structure Image Recognition

train / test datasets : DECIMER

github 디렉토리 구조

RePaint

guided-diffusion

MolScribe

## README.md

codebase

link[MolScribe](https://github.com/thomas0809/MolScribe.git)

link[RePaint](https://github.com/andreas128/RePaint.git)

link[guided-diffusion](https://github.com/openai/guided-diffusion.git)

## RePaint

### environment 세팅

```bash
cd RePaint
conda create -n repaint python
conda activate repaint
pip install -e .
pip install numpy torch blobfile tqdm pyYaml pillow # Example: torch 1.7.1+cu110.
conda install -c conda-forge mpi4py
conda install -c conda-forge openmpi
```

### DDPM train

```bash
mkdir ddpm_train_log

export OPENAI_LOGDIR=ddpm_train_log

MODEL_FLAGS=“—image_size 256 —attention_resolutions “32, 16, 8” —num_channerls 256 —num_head_channels 64 —num_res_blocks 2 —num_heads 4 —resblock_updown true —learn_sigma true —use_scale_shift_norm true —timestep_respacing “250”  —use_kl false —class_cond false —dropout 0.0“

DIFFUSION_FLAGS=“diffusion_steps 1000 —noise_schedule “linear” —rescale_learned sigmas false "

TRAIN_FLAGS=“—use_fp16 false  —batch_size 32 —microbatch 1 —lr 2e-5 —log_interval 100 —save_interval 10000”
```

```bash
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Note: export OPENAI_LOGDIR sets the location for DDPM training logs. If not set, logs are saved in the /tmp folder. The model checkpoints are saved at intervals specified by save_interval and are named in the format opt_0.999_{step}, ema_0.999_{step}, model_0.999_{step}. Use the ema prefixed file for sampling. Training does not have a predefined maximum step; it should be determined based on the quality of image samples from the checkpoints. The checkpoint for the model used in the paper is ema_0.999_210000.pt.

### RePaint

Configuration files are located in the confs folder.

Example configuration in molecule.yml:

```yaml
name: molecule_example
data:
  eval:
    paper_face_mask:
      mask_loader: true
      gt_path: ./data/datasets/gts/face
      mask_path: ./data/datasets/gt_keep_masks/face
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
      paths:
        srs: ./log/face_example/inpainted
        lrs: ./log/face_example/gt_masked
        gts: ./log/face_example/gt
        gt_keep_masks: ./log/face_example/gt_keep_mask
```

After customizing name, gt_path, mask_path, and paths (srs, lrs, gts, gt_keep_masks):

custom하고 저장한 후에

```bash
python test.py --conf_path confs/molecule_example.yml
```

Sampled files are saved in ./log/molecule_example/inpainted.

### MolScribe

### environment 세팅

```bash
cd MolScribe
conda env create -f environment.yml
conda activate molscribe
mkdir -p ckpts
wget -P ckpts https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth
```

## Data Files

# data
1. filterered_DECIMER_train_train.csv - original train data
2. filterered_DECIMER_train_val.csv - original validation data
3. transform_rdkit.csv - original + rdkit(image Augmentation)
4. transform_randepict.csv - original + randepict(image Augmentation)
5. transform_repaint.csv - original + repaint(image Augmentation)
6. s_repaint_horizontal.csv - horizontal(image, SMILES Augmentation)
7. a_repaint_vertical.csv - vertical(image, SMILES Augmentation)
6. filterered_DECIMER_3194_test.csv - original test data

# image_dir
1. filterered_DECIMER_3194_train
2. filterered_DECIMER_3194_test
3. rdkit_image
4. Randepict_DECIMER_train_sets_OCSR
5. molecule_horizontal_example_1
6. molecule_horizontal_example_2
7. molecule_vertical_example_1 
8. molecule_vertical_example_2

MolScribe 훈련에 사용할 데이터 형식

csv파일 구성요소

columns 

image_id : image_identifier

file_path : image_id의 image file path

SMILES : image_id에 해당하는 SMILES

## Train

```bash
bash train_scripts/s_repaint.sh
```

sh파일 수정 옵션

NUM_GPUS_PER_NODE : GPU 개수

BATCH_SIZE : batch size

SAVE_PATH : 체크포인트 파일 및 validation sets에 대한 predict 파일 저장 경로

train_file : train data path

valid_file : validation data path

epoch : fine-tuning epoch 수

fine-tuning이 아닌 처음부터 훈련을 원한다면 MolScribe 참고하면 된다.

## Predict

```bash
bash pred_scripts/predict.sh
```

single checkpoint predict - single_predict.sh

Iterative checkpoint predict - multi_predict.sh

## Evaluate

```bash
bash evaluate_scripts/single_eval.sh
```

```bash
bash evaluate_scripts/eval.sh
```

[eval.sh](http://eval.sh) 파일에서

Customize the BASE_DIR, GOLD_FILE, and PRED_PATH as needed in the evaluation scripts.

single evaluate - single_eval.sh

Iterative evaluate - eval.sh
