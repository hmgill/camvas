import pathlib

project_root = pathlib.Path(__file__).parent.absolute()

project_paths = {
    "root": project_root,
    "output": pathlib.Path(project_root, "output"),
    "predictions": pathlib.Path(project_root, "output", "predictions"),
    "snapshot_dill": pathlib.Path(project_root, "output", "snapshots", "dill"),
    "snapshot_plots": pathlib.Path(project_root, "output", "snapshots", "plots"),
    "history": pathlib.Path(project_root, "core", "training", "history"),
    "weights": pathlib.Path(project_root, "weights"),
    "visualization": pathlib.Path(project_root, "visualization"),
    "scripts": pathlib.Path(project_root, "scripts")
}
