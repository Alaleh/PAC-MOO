import os
from benchmarks import *


def get_objective_function(name):
    if name == "OSY":
        return OSY
    else:
        raise "Unimplemented objective function error"


def write_to_file(data, paths, file_path):
    with open(os.path.join(paths, file_path), "a") as filehandle:
        for item in data:
            filehandle.write('%f ' % item)
        filehandle.write('\n')
    filehandle.close()
