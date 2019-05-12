################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

## Added FAST_MLPG as a variable here, in case someone wants to use the slow version, but perhaps we
## should always use the bandmat version?
FAST_MLPG = True
#io_funcs.

from binary_io import  BinaryIOCollection
import os, re, numpy
import logging
import argparse

from scipy import signal

if FAST_MLPG:
    from mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
#    pass
else:
    from mlpg import MLParameterGeneration

class   ParameterGeneration(object):

    def __init__(self, gen_wav_features = ['mgc', 'lf0'], enforce_silence=False):
        self.gen_wav_features = gen_wav_features
        self.enforce_silence  = enforce_silence

        # Debug:
        self.inf_float = -1.0e+10

        self.var = {}

    def duration_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict):

        logger = logging.getLogger('param_generation')

        logger.debug('duration_decomposition for %d files' % len(in_file_list) )

        state_number = 5  ## hard coding, try removing in future?

        if len(list(out_dimension_dict.keys()))>1:
            logger.critical("we don't support any additional features along with duration as of now.")
            sys.exit(1)
        else:
            feature_name = list(out_dimension_dict.keys())[0]

        io_funcs = BinaryIOCollection()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:

            findex=findex+1

            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            gen_features = numpy.int32(numpy.round(features))
            gen_features[gen_features<1]=1

            if dimension > state_number:
                gen_features = gen_features[:, state_number]

            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])
            io_funcs.array_to_binary_file(gen_features, new_file_name)

            logger.debug('wrote to file %s' % new_file_name)

    def acoustic_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict, var_file_dict, do_MLPG=True, cfg=None):

        print ('param_generation')

        print ('acoustic_decomposition for %d files' % len(in_file_list) )

        self.load_covariance(var_file_dict, out_dimension_dict)

        stream_start_index = {}
        dimension_index = 0
        recorded_vuv = True
        vuv_dimension = None

        for feature_name in list(out_dimension_dict.keys()):
            if feature_name != 'vuv':
               stream_start_index[feature_name] = dimension_index
            else:
               vuv_dimension = dimension_index
            dimension_index += out_dimension_dict[feature_name]

        io_funcs = BinaryIOCollection()

        mlpg_algo = MLParameterGeneration()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:

            findex=findex+1

            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)

            print('processing %4d of %4d: %s' % (findex,flen,file_name) )

            for feature_name in self.gen_wav_features:

                print(' feature: %s' % feature_name)
                current_features = features[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]
                if FAST_MLPG:
                    ### fast version wants variance per frame, not single global one:
                    var = self.var[feature_name]
                    var = numpy.transpose(numpy.tile(var,frame_number))
                else:
                    var = self.var[feature_name]

#                print  var.shape[1]
                if do_MLPG == False:
                    gen_features = current_features
                else:
                    gen_features = mlpg_algo.generation(current_features, var, out_dimension_dict[feature_name]//3)

                print(' feature dimensions: %d by %d' %(gen_features.shape[0], gen_features.shape[1]))

                if feature_name in ['lf0', 'F0']:
                    if 'vuv' in stream_start_index:
                        vuv_feature = features[:, stream_start_index['vuv']:stream_start_index['vuv']+1]

                        for i in range(frame_number):
                            if new_vuv_feature[i] < 0.5:
                                gen_features[i, 0] = self.inf_float

                new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])

                if self.enforce_silence:
                    silence_pattern = cfg.silence_pattern
                    label_align_dir = cfg.in_label_align_dir
                    in_f = open(label_align_dir+'/'+file_id+'.lab','r')
                    for line in in_f.readlines():
                        line = line.strip()

                        if len(line) < 1:
                            continue
                        temp_list  = re.split('\s+', line)
                        start_time = int(int(temp_list[0])*(10**-4)/5)
                        end_time   = int(int(temp_list[1])*(10**-4)/5)

                        full_label = temp_list[2]

                        label_binary_flag = self.check_silence_pattern(full_label, silence_pattern)

                        if label_binary_flag:
                            if feature_name in ['lf0', 'F0', 'mag']:
                                gen_features[start_time:end_time, :] = self.inf_float
                            else:
                                gen_features[start_time:end_time, :] = 0.0

                io_funcs.array_to_binary_file(gen_features, new_file_name)
                print(' wrote to file %s' % new_file_name)


    def load_covariance(self, var_file_dict, out_dimension_dict):

        io_funcs = BinaryIOCollection()
        for feature_name in list(var_file_dict.keys()):
            var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

            var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

            self.var[feature_name] = var_values


    def check_silence_pattern(self, label, silence_pattern):
        for current_pattern in silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="exp/acoustic/test")
    parser.add_argument("--var_dir", default="var")
    args=parser.parse_args()

    cmp_dir = os.path.join(args.out_dir, "cmp")
    mgc_dir = os.path.join(args.out_dir, "mgc")
    lf0_dir = os.path.join(args.out_dir, "lf0")

    in_file_list = []
    for fn in sorted(os.listdir(cmp_dir)):
        in_file_list.append(os.path.join(cmp_dir, fn))
    out_dimension_dict = { 'mgc' : 123,
                           'vuv' : 1,
			   'lf0' : 3}

    file_extension_dict = {'mgc' : '.mgc',
			   'lf0' : '.lf0',
                           'vuv' : '.vuv'}

    var_file_dict  = {'mgc':'{}/mgc.var'.format(args.var_dir),
		      'vuv':'{}/vuv.var'.format(args.var_dir),
                      'lf0':'{}/lf0.var'.format(args.var_dir)}

    generator = ParameterGeneration()

    # out_dimension_dict is the cmp structrue of your nnet output cmp
    generator.acoustic_decomposition(in_file_list, 127, out_dimension_dict, file_extension_dict, var_file_dict)

    if not os.path.exists(lf0_dir):
        os.mkdir(lf0_dir)
    if not os.path.exists(mgc_dir):
        os.mkdir(mgc_dir)
    os.system('mv {}/*.lf0 {}'.format(cmp_dir, lf0_dir))
    os.system('mv {}/*.mgc {}'.format(cmp_dir, mgc_dir))

    io_funcs = BinaryIOCollection()
    inf_float = -1.0e+10
    for item in os.listdir(cmp_dir):
        vuv = numpy.reshape(numpy.fromfile(os.path.join(cmp_dir, item), dtype=numpy.float32), [-1,127])[:,123]
        name, ext = os.path.splitext(item)
        lf0 = numpy.reshape(numpy.fromfile(os.path.join(lf0_dir, "{}.lf0".format(name)),dtype=numpy.float32),[-1,1])
        mgc = numpy.reshape(numpy.fromfile(os.path.join(mgc_dir, "{}.mgc".format(name)),dtype=numpy.float32),[-1,41])
        mgc = signal.convolve2d(
            mgc, [[1.0 / 3], [1.0 / 3], [1.0 / 3]], mode="same", boundary="symm")
        lf0[vuv < 0.5] = inf_float
        io_funcs.array_to_binary_file(lf0, os.path.join(lf0_dir, "{}.lf0".format(name)))
        io_funcs.array_to_binary_file(mgc, os.path.join(mgc_dir, "{}.mgc".format(name)))
