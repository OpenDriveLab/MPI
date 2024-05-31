CUDA_VISIBLE_DEVICES=0 python mpi_evaluation/referring_grounding/evaluate_refer.py test_only=False iou_threshold=0.5 lr=1e-3 \
load_checkpoint=\"\" \
model=\"mpi-small\" \
save_path=\"MPI-Small-IOU0.5\" \
eval_checkpoint_path=\"MPI-small-state_dict.pt\" \
language_model_path=\"distilbert-base-uncased\" \

