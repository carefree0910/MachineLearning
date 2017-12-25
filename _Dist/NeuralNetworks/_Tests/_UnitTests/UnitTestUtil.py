import os
import shutil

root_cwd = os.path.abspath("../")


def clear_cache():
    cwd = os.getcwd()
    local_data_folder = os.path.join(cwd, "_Data")
    if os.path.isdir(local_data_folder):
        shutil.rmtree(local_data_folder)
    shutil.rmtree(os.path.join(cwd, "_Models"))
    remote_cache_folder = os.path.join(root_cwd, "_Data", "_Cache")
    remote_info_folder = os.path.join(root_cwd, "_Data", "_DataInfo")
    if os.path.isdir(remote_cache_folder):
        shutil.rmtree(remote_cache_folder)
    if os.path.isdir(remote_info_folder):
        shutil.rmtree(remote_info_folder)
