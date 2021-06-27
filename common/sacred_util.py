from sacred.observers import FileStorageObserver

def get_run_path(_run):
    file_obs_list = [o for o in _run.observers if isinstance(o, FileStorageObserver)]
    if file_obs_list:
        run_path = file_obs_list[0].dir
    else:
        run_path = 'Figures'
    return run_path

