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


import sys, numpy,os
sys.path.append('/home/work_nfs2/jcong/workspace/merlin/src')
import io_funcs
from io_funcs.binary_io import BinaryIOCollection
import  logging
from scipy.stats.stats import pearsonr

class   DistortionComputation(object):
    def __init__(self):
        self.total_frame_number = 0
        self.distortion = 0.0
        self.bap_distortion = 0.0
        self.f0_distortion = 0.0
        self.vuv_error = 0.0

        self.mgc_dim = 60
        self.bap_dim = 5
        self.lf0_dim = 1

    def compute_distortion(self, file_id_list, reference_dir, generation_dir):
        total_voiced_frame_number = 0
        file_cnt = 0
        for file_id in open(file_id_list,'r').readlines():
            file_id =file_id.strip()
            file_cnt += 1

            mgc_file_name0 = reference_dir + '/' + 'mgc' + '/' + file_id + '.mgc'
            bap_file_name0 = reference_dir + '/' + 'bap' + '/' + file_id + '.bap'
            lf0_file_name0 = reference_dir + '/' + 'lf0' + '/' + file_id + '.lf0'

            mgc_file_name = generation_dir + '/' + 'mgc'  + '/' + file_id + '.mgc'
            bap_file_name = generation_dir + '/' + 'bap'  + '/' + file_id + '.bap'
            lf0_file_name = generation_dir + '/' + 'lf0'  + '/' + file_id + '.lf0'

            generation_mgc, mgc_frame_number = self.load_binary_file(mgc_file_name, self.mgc_dim)
            generation_lf0, lf0_frame_number = self.load_binary_file(lf0_file_name, self.lf0_dim)
            generation_bap, bap_frame_number = self.load_binary_file(bap_file_name, self.bap_dim)

            reference_mgc, mgc_frame_number0 = self.load_binary_file(mgc_file_name0, self.mgc_dim)
            reference_lf0, lf0_frame_number0 = self.load_binary_file(lf0_file_name0, self.lf0_dim)
            reference_bap, bap_frame_number0 = self.load_binary_file(bap_file_name0, self.bap_dim)


            if mgc_frame_number0 != mgc_frame_number:
                print("The number of mgc frames is not the same: %d vs %d. Error in compute_distortion.py" %(mgc_frame_number0, mgc_frame_number))
                if mgc_frame_number0 < mgc_frame_number:
                    generation_mgc = generation_mgc[0:mgc_frame_number0]
                    mgc_frame_number = mgc_frame_number0
                else:
                    reference_mgc = reference_mgc[0:mgc_frame_number]
                    mgc_frame_number0 = mgc_frame_number
                print('Ignore the redundant frames')

            if lf0_frame_number0 != lf0_frame_number:
                print("The number of lf0 frames is not the same: %d vs %d. Error in compute_distortion.py" %(lf0_frame_number0, lf0_frame_number))
                if lf0_frame_number0 < lf0_frame_number:
                    generation_lf0 = generation_lf0[0:lf0_frame_number0]
                    lf0_frame_number = lf0_frame_number0
                else:
                    reference_lf0 = reference_lf0[0:lf0_frame_number]
                    lf0_frame_number0 = lf0_frame_number
                print('Ignore the redundant frames')

            temp_distortion = self.compute_mse(reference_mgc[:, 1:self.mgc_dim], generation_mgc[:, 1:self.mgc_dim])
            self.distortion += temp_distortion * (10 /numpy.log(10)) * numpy.sqrt(2.0)

            temp_bap_distortion = self.compute_mse(reference_bap, generation_bap)
            self.bap_distortion += temp_bap_distortion * (10 /numpy.log(10)) * numpy.sqrt(2.0)

            temp_f0_distortion, temp_vuv_error, voiced_frame_number = self.compute_f0_mse(reference_lf0, generation_lf0)
            self.f0_distortion += temp_f0_distortion
            self.vuv_error += temp_vuv_error

            self.total_frame_number += mgc_frame_number0
            total_voiced_frame_number += voiced_frame_number

        self.distortion /= float(self.total_frame_number)
        self.bap_distortion /= float(self.total_frame_number)

        self.f0_distortion /= total_voiced_frame_number
        self.f0_distortion = numpy.sqrt(self.f0_distortion)

        self.vuv_error /= float(self.total_frame_number)
        print('---------------------------------------------------------------------')
        print('Total file number is: %d' % (file_cnt))
        print('MCD (MGC Distortion):  %.3f dB' %(self.distortion))
        print('BAPD (BAP Distortion): %.3f dB' %(self.bap_distortion))
        print('RMSE (RMSE in lof_f0): %.3f Hz' %(self.f0_distortion))
        print('VUV (V/UV Error Rate): %.3f%%' % (self.vuv_error*100.))
        return  self.distortion, self.bap_distortion, self.f0_distortion, self.vuv_error

    def compute_f0_mse(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        voiced_frame_number = voiced_gen_data.size

        f0_mse = numpy.sum(((numpy.exp(voiced_ref_data) - numpy.exp(voiced_gen_data)) ** 2))
        # f0_mse = numpy.sum((((voiced_ref_data) - (voiced_gen_data)) ** 2))

        vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        vuv_error = numpy.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

        return  f0_mse, vuv_error, voiced_frame_number

    def compute_mse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        sum_diff = numpy.sum(diff, axis=1)
        sum_diff = numpy.sqrt(sum_diff)
        sum_diff = numpy.sum(sum_diff, axis=0)
        return  sum_diff

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        frame_number = features.size / dimension
        features = features.reshape((-1, dimension))
        return  features, frame_number


'''
to be refined. genertic class for various features
'''
class IndividualDistortionComp(object):

    def __init__(self):
        self.logger = logging.getLogger('computer_distortion')

    def compute_distortion(self, file_id_list, reference_dir, generation_dir, file_ext, feature_dim):
        total_voiced_frame_number = 0

        distortion = 0.0
        vuv_error = 0
        total_frame_number = 0

        io_funcs = BinaryIOCollection()

        ref_all_files_data = numpy.reshape(numpy.array([]), (-1,1))
        gen_all_files_data = numpy.reshape(numpy.array([]), (-1,1))

        fout = open(file_id_list,'r')

        for file_id in fout.readlines():
            file_id =file_id.strip()
            # print file_id
            ref_file_name  = reference_dir + '/' + file_id + file_ext
            gen_file_name  = generation_dir + '/' + file_id + file_ext

            ref_data, ref_frame_number = io_funcs.load_binary_file_frame(ref_file_name, feature_dim)
            gen_data, gen_frame_number = io_funcs.load_binary_file_frame(gen_file_name, feature_dim)

            if ref_frame_number != gen_frame_number:
                if ref_frame_number > gen_frame_number:
                    ref_data = ref_data[:gen_frame_number]
                    ref_frame_number = gen_frame_number
                else:
                    gen_data = gen_data[:ref_frame_number]
                    gen_frame_number = ref_frame_number


            if file_ext == '.lf0':
                ref_all_files_data = numpy.concatenate((ref_all_files_data, ref_data), axis=0)
                gen_all_files_data = numpy.concatenate((gen_all_files_data, gen_data), axis=0)
                temp_distortion, temp_vuv_error, voiced_frame_number = self.compute_f0_mse(ref_data, gen_data)
                vuv_error += temp_vuv_error
                total_voiced_frame_number += voiced_frame_number
            elif file_ext == '.dur':
                ref_data = numpy.reshape(numpy.sum(ref_data, axis=1), (-1, 1))
                gen_data = numpy.reshape(numpy.sum(gen_data, axis=1), (-1, 1))
                ref_all_files_data = numpy.concatenate((ref_all_files_data, ref_data), axis=0)
                gen_all_files_data = numpy.concatenate((gen_all_files_data, gen_data), axis=0)
                continue;
            elif file_ext == '.mgc':
                temp_distortion = self.compute_mse(ref_data[:, 1:feature_dim], gen_data[:, 1:feature_dim])
            else:
                temp_distortion = self.compute_mse(ref_data, gen_data)

            distortion += temp_distortion

            total_frame_number += ref_frame_number

        if file_ext == '.dur':
            dur_rmse = self.compute_rmse(ref_all_files_data, gen_all_files_data)
            dur_corr = self.compute_corr(ref_all_files_data, gen_all_files_data)

            return dur_rmse, dur_corr
        elif file_ext == '.lf0':
            distortion /= float(total_voiced_frame_number)
            vuv_error  /= float(total_frame_number)

            distortion = numpy.sqrt(distortion)
            f0_corr = self.compute_f0_corr(ref_all_files_data, gen_all_files_data)
            print('F0_Corr ((F0_Correlation): %.3f' % (f0_corr))
            print('---------------------------------------------------------------------')
            return  distortion, f0_corr, vuv_error
        else:
            distortion /= float(total_frame_number)

            return  distortion

    def compute_f0_mse(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        voiced_frame_number = voiced_gen_data.size

        f0_mse = (numpy.exp(voiced_ref_data) - numpy.exp(voiced_gen_data)) ** 2
        f0_mse = numpy.sum((f0_mse))

        vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        vuv_error = numpy.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

        return  f0_mse, vuv_error, voiced_frame_number

    def compute_f0_corr(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        f0_corr = self.compute_corr(numpy.exp(voiced_ref_data), numpy.exp(voiced_gen_data))
        return f0_corr

    def compute_corr(self, ref_data, gen_data):
        corr_coef = pearsonr(ref_data, gen_data)

        return corr_coef[0]

    def compute_rmse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        total_frame_number = ref_data.size
        sum_diff = numpy.sum(diff)
        rmse = numpy.sqrt(sum_diff/total_frame_number)

        return rmse

    def compute_mse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        sum_diff = numpy.sum(diff, axis=1)
        sum_diff = numpy.sqrt(sum_diff)       # ** 0.5
        sum_diff = numpy.sum(sum_diff, axis=0)
        return  sum_diff

def main(file_id_list, reference_dir, generation_dir):
    dis_cmp = DistortionComputation()
    dis_cmp.compute_distortion(file_id_list, reference_dir, generation_dir)

    reference_lf0_dir = os.path.join(reference_dir,'lf0')
    generation_lf0_dir = os.path.join(generation_dir,'lf0')
    invd = IndividualDistortionComp()
    invd.compute_distortion(file_id_list, reference_lf0_dir, generation_lf0_dir, ".lf0", 1)

if __name__ == '__main__':
    file_id_list = sys.argv[1]
    reference_dir = sys.argv[2]
    generation_dir = sys.argv[3]
    main(file_id_list, reference_dir, generation_dir)
