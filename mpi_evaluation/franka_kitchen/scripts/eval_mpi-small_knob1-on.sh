CUDA_VISIBLE_DEVICES=0 PYTHONPATH=mpi_evaluation/franka_kitchen/MPIEval/core python mpi_evaluation/franka_kitchen/MPIEval/core/hydra_launcher.py hydra/launcher=local hydra/output=local env="kitchen_knob1_on-v3" camera="left_cap2" pixel_based=true embedding=ViT-Small num_demos=25 env_kwargs.load_path=mpi-small bc_kwargs.finetune=false job_name=mpi-small seed=125 proprio=9