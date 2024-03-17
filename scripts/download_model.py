import questionary
import wandb

from arithmetic_lm.constants import CHECKPOINTS_DIR

WANDB_ENTITY = "compositional-generalization-ut"

if __name__ == "__main__":
    api = wandb.Api()

    # prep a list of runs
    runs = []
    for proj in api.projects(WANDB_ENTITY):
        proj_name = f"{WANDB_ENTITY}/{proj.name}"
        for run in api.runs(proj_name):
            if run.state == "finished":
                runs.append(run)

    # ask the user to select runs to download models
    selected_runs = questionary.checkbox(
        "Select runs to download models",
        choices=[
            questionary.Choice(title=f"{run.project}/{run.name}", value=run)
            for run in runs
        ],
    ).ask()

    # download the models
    for run in selected_runs:
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                out_dir = CHECKPOINTS_DIR / run.project / run.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "model.ckpt"
                print(
                    f"Downloading {artifact.name} from {run.project}/{run.name} to {out_path}"
                )
                artifact.download(out_path)
