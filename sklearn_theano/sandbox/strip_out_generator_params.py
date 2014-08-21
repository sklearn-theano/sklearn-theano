from sklearn.externals.joblib import dump
import sys
from pylearn2.utils import serial

model = serial.load(sys.argv[1])
mlp = model.generator.mlp
dump(mlp.get_param_values(), 'generator_params.jb')
