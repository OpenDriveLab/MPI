
# Learning Manipulation by Predicting Interaction (MPI)


<h3 align="center">
  <a href="https://opendrivelab.github.io/mpi.github.io/#">Project Website</a> |
  <a href="https://opendrivelab.github.io/mpi.github.io/resources/RSS_2024_MPI.pdf">Paper</a> |
  RSS 2024
</h3>

<img width="1000" alt="mpi" src="assets/mpi_teaser.png">

## :fire: Highlight

​**MPI** is an interaction-oriented representation learning method towards robot manipulation:

- Instruct the model towards predicting transition frames and detecting manipulated objects with keyframes.
- Foster better comprehension of “how-to-interact” and “where-to-interact”.
- Acquire more informative representations during pre-training and achieve evident improvement across downstream tasks.


## :rocket: News

- **[2024/05/31]** We released the implementation of pre-training and evaluation on Referring Expression Grounding task. 

## :page_facing_up: TODO List

- [ ] Model weights release.
- [ ] Evaluation code on Franka Kitchen environment.


## Getting Started <a name="start"></a>
- [Installation](#installation)
- [Prepare Dataset](#dataset)
- [Pre-training](#pretraining)
- [Evaluation](#evaluation)

### Installation <a name="installation"></a>

Step 1. Clone and setup MPI dependency:
```bash
git clone https://github.com/OpenDriveLab/MPI
cd MPI
pip install -e .
```

Step 2. Prepare the language model, you may download DistillBERT from [HuggingFace](https://huggingface.co/distilbert/distilbert-base-uncased)


### Prepare Dataset <a name="dataset"></a>
Download [Ego4D](https://ego4d-data.org/docs/start-here/) Hand-and-Object dataset:
```
# Download the CLI
pip install ego4d
# Select Subset Of Hand-and-Object
python -m ego4d.cli.cli --output_directory=<path-to-save-dir> --datasets clips annotations  --metadata --version v2 --benchmarks FHO
```
Your directory tree should look like this: 
```
$<path-to-save-dir>
├── ego4d.json
└── v2
    |—— annotations  
    └── clips
```
Preprocess dataset for pre-training MPI:
```
python prepare_dataset.py --root_path <path-to-save-dir>/v2/
```

### Pre-training <a name="pretraining"></a>

<img width="1000" alt="mpi" src="assets/pretrain_pipeline.png">

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 pretrain.py
```

### Evaluation <a name="evaluation"></a>
#### Referring Expression Grounding
Step 1. Prepare the OCID-Ref dataset following this [repo](https://github.com/lluma/OCID-Ref). Then put the dataset to 

```bash
/mpi_evaluation/referring_grounding/data/langref
```

Step 2. Initiate evaluation with
```bash
python mpi_evaluation/referring_grounding/evaluate_refer.py test_only=False iou_threshold=0.5 lr=1e-3 \
model=\"mpi-small\" \
save_path=\"MPI-Small-IOU0.5\" \
eval_checkpoint_path=\"path_to_your/MPI-small-state_dict.pt\" \
language_model_path=\"path_to_your/distilbert-base-uncased\" \
```

or you can simply use 
```bash
bash mpi_evaluation/referring_grounding/eval_refer.sh
```

#### Franka Kitchen
**TBD**

## Citation

If you find the project helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{zeng2024mpi,
  title={Learning Manipulation by Predicting Interaction},
  author={Jia, Zeng and Qingwen, Bu and Bangjun, Wang and Wenke, Xia and Li, Chen and Hao, Dong and Haoming, Song and Dong, Wang and Di, Hu and Ping, Luo and Heming, Cui and Bin, Zhao and Xuelong, Li and Yu, Qiao and Hongyang, Li},
  booktitle= {Proceedings of Robotics: Science and Systems (RSS)},
  year={2024}
}
```

## Acknowledgment
The code of this work is built upon [Voltron](https://github.com/siddk/voltron-robotics) and [R3M](https://github.com/facebookresearch/r3m). Thanks for their open-source work!
