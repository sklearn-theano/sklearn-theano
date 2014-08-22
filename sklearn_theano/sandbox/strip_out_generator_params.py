import os
import sys
from pylearn2.utils import serial
from sklearn.externals.joblib import dump

#This line is meant for having this script in galatea/adversarial
if os.path.exists('../../galatea'):
    sys.path.append('../../')
model = serial.load(sys.argv[1])
mlp = model.generator.mlp
dump(mlp.get_param_values(), 'generator_params.jb')
