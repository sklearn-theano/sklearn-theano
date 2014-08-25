import numpy as np
import Image
import sys
import os
import StringIO
import subprocess


def get_overfeat_dir(overfeat_dir=None):
    if overfeat_dir is None:
        overfeat_dir = os.environ.get("OVERFEAT_DIR", None)
    return overfeat_dir


def get_overfeat_cmd(overfeat_cmd=None, overfeat_dir=None, architecture=None):

    if overfeat_cmd is None:
        overfeat_cmd = os.environ.get("OVERFEAT_CMD", None)
    if overfeat_cmd is None:
        overfeat_dir = get_overfeat_dir(overfeat_dir)
        if overfeat_dir is None or architecture is None:
            raise Exception(
                'Please set the environment variable OVERFEAT_CMD'
                ' to point to the file /path/to/overfeat/bin/(system'
                ')/overfeatcmd, or set OVERFEAT_DIR and provide architecture')
        overfeat_cmd = os.path.join(overfeat_dir, "bin", architecture,
                                    "overfeatcmd")
    return overfeat_cmd


def  get_net_weight_dir(net_weight_dir=None, overfeat_dir=None):
    if net_weight_dir is None:
        net_weight_dir = os.environ.get("OVERFEAT_NET_WEIGHT_DIR", None)
    if net_weight_dir is None:
        overfeat_dir = get_overfeat_dir(overfeat_dir)
        if overfeat_dir is not None:
            net_weight_dir = os.path.join(overfeat_dir, "data/default/")
        else:
            raise Exception("Please provide net_weight_dir or  set "
                            "OVERFEAT_NET_WEIGHT_DIR or OVERFEAT_DIR")
    return net_weight_dir


def get_net_weights(net_weight_file=None, large_net=None,
                    net_weight_dir=None, overfeat_dir=None):
    if net_weight_file is None:
        net_weight_file = os.environ.get("OVERFEAT_NET_WEIGHT_FILE", None)
    if net_weight_file is None:
        net_weight_dir = get_net_weight_dir(net_weight_dir, overfeat_dir)
        if large_net is not None:
            net_weight_file = os.path.join(net_weight_dir, "net_weight_%d"
                                           % int(large_net))
        else:
            raise Exception("Please specify large_net=0/1/False/True")
    return net_weight_file


def get_overfeat_output_raw(img_arr, layer_id, largenet, overfeatcmd=None,
                            net_weight_file=None, overfeat_dir=None,
                            architecture='linux_64'):

    if img_arr.dtype != np.uint8:
        raise ValueError('Please convert image to uint8')

    if img_arr.shape[2] != 3:
        raise ValueError('Last dimension must index color')

    overfeatcmd = get_overfeat_cmd(overfeatcmd, overfeat_dir, architecture)
    net_weight_file = get_net_weights(net_weight_file, largenet,
                                      overfeat_dir=overfeat_dir)

    image = Image.fromarray(img_arr)

    buf = StringIO.StringIO()
    image.save(buf, format='ppm')
    buf.seek(0)

    command = overfeatcmd + " " + net_weight_file + " -1 %d %d" % (
        int(largenet), layer_id)

    p = subprocess.Popen(
        command.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output = p.communicate(input=buf.buf)[0]

    return output


def _parse_overfeat_output(output):
    buf = StringIO.StringIO(output)
    shape_ = buf.readline()
    content_ = buf.readline()

    shape = map(int, shape_.strip().split(" "))
    content = [map(float, content_.strip().split(" "))]

    return np.array(content).reshape(shape)


def get_output(img_arr, layer_id, largenet, overfeatcmd=None,
               net_weight_file=None, overfeat_dir=None,
               architecture='linux_64'):

    output = get_overfeat_output_raw(img_arr, layer_id, largenet,
                                     overfeatcmd, net_weight_file,
                                     overfeat_dir, architecture)
    return _parse_overfeat_output(output)

