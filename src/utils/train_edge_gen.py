import os
import sys
import wandb

from edgegen_utils import *

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    wandb.init(
        project="Object-Removal-Edge-Gen", 
        name="Finetune-Places2-Run2", 
        dir=project_root, 
        config={
            "learning_rate": 1e-5,
            "architecture": "EdgeConnect",
            "dataset": "Places2 Subset",
            "epochs": 20,
            "batch_size": 8 
        }
    )
    train_edge_generator("/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/edge_finetuned_epoch_4.pth", batch_size=8, start_ep=4)