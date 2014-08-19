import numpy as np
try:
    import Image
except:
    from PIL import Image
import os
try:
    import commands
except:
    import subprocess as commands
import tempfile

overfeat_binary = os.environ.get('OVERFEAT_BINARY', None)
if overfeat_binary is None:
    raise Exception("Please provide overfeat command in OVERFEAT_BINARY env"
                    "ironment variable. It is located in path/to/overfeat/"
                    "bin/linux64/overfeat")

ld_library_path = os.environ.get('LD_LIBRARY_PATH', None)
if ld_library_path is None:
    import warnings
    warnings.warn("You may need to set LD_LIBRARY_PATH to your OpenBlas"
                  "lib directory e.g /usr/lib/openblas-base")


def _jpg_or_png(image, tmp_file_dir=None, vmin=None, vmax=None):
    """Helper to make sure input is an image file. Will create a temporary
    file if necessary"""

    if isinstance(image, str):
        if image[-4:] == '.png':
            return image
        elif image[-4:] == '.jpg':
            return image
        else:
            raise NotImplementedError("Add filename suffixes here")

    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            vmin = vmin or image.min()
            vmax = vmax or image.max()
            image = (image - vmin) / float(vmax - vmin)
            image = image * 256
            image = image.astype(int)
            image[image == 256] = 255
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = image[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                      dtype=np.uint8)

        prefix = "overfeat"
        img = Image.fromarray(image)
        f = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png",
                                        dir=tmp_file_dir, delete=False)

        img.save(f)
        f.close()
        return f.name
    else:
        raise ValueError("Don't understand input image")


def get_overfeat_layer(image, layer=None, large_net=False, n_top=None):

    layer_cmd = (layer is not None and " --features-layer=%d" % layer) or ""
    large_cmd = (large_net and " --large_net") or ""
    n_top_cmd = (n_top is not None and " -n %d" % n_top) or ""

    image = _jpg_or_png(image)

    command = overfeat_binary + layer_cmd + large_cmd + n_top_cmd + " " \
        + image

    output = commands.getoutput(command)
    shape, content = output.split("\n")
    shape = list(map(int, shape.split(" ")))
    content = list(map(float, content.strip().split(" ")))

    output = np.array(content).reshape(shape)

    return output


if __name__ == "__main__":
    from scipy.misc import lena

    out = get_overfeat_layer(lena(), layer=3)

