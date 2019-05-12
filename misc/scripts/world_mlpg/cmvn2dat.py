import os
import numpy as np
from binary_io2 import BinaryIOCollection
import argparse

class AcousticCMVN(object):

    def var_write(self, var_dir, std_vector, out_dimension_dict):
        out_std_vector = np.zeros((1,len(std_vector)))
        out_std_vector[0,:] = std_vector
        feature_index = 0
        for feature_name in list(out_dimension_dict.keys()):
                    feature_std_vector = np.array(out_std_vector[:,feature_index:feature_index+out_dimension_dict[feature_name]], 'float32')
                    io_funcs = BinaryIOCollection()
                    io_funcs.array_to_binary_file(feature_std_vector,os.path.join(var_dir, feature_name+".var"))
                    feature_index += out_dimension_dict[feature_name]

    def caculate_cmvn(self, FLAGS, dim, out_dimension_dict):
        io_funcs = BinaryIOCollection()
        cmvn = np.load(FLAGS.cmvn)
        stddev = cmvn["stddev_labels"]
        var = pow(stddev, 2)
        self.var_write(FLAGS.var, var, out_dimension_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--var',
        type =str,
        default = 'data/var',
        help ='output file,default:var'
        )
    parser.add_argument(
        '--cmvn',
        type =str,
        default = 'data/train_cmvn.npz',
        help ='output file, default:var'
        )
    FLAGS,unparser = parser.parse_known_args()
    acoustic_cmvn = AcousticCMVN()
    if not os.path.exists(FLAGS.var):
        os.mkdir(FLAGS.var)
    out_dimension_dict = {'mgc' : 180,
                          'lf0' : 3,
                          'vuv' : 1,
                          'bap' : 15}
    acoustic_cmvn.caculate_cmvn(FLAGS, 199, out_dimension_dict)
