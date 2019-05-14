#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2018 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npujcong@gmail.com (congjian)

import sys
import os
import struct
import shutil
import numpy as np
import argparse
from scipy import signal

def write_binary_file(data, output_file_name, with_dim=False):
    data = np.asarray(data, np.float32)
    fid = open(output_file_name, 'wb')
    if with_dim:
        fid.write(struct.pack('<i', data.shape[0]))
        fid.write(struct.pack('<i', data.shape[1]))
    data.tofile(fid)
    fid.close()

def pz_spiltcmp(f_cmp, f_mgc, f_lf0):
    inf_float = -1.0e+10
    with open(f_cmp,'rb') as fid:
        data = np.fromfile(fid,dtype=np.float32)
        data = np.reshape(data,[-1,51])
    mgc = signal.convolve2d(
        data[:, 0:41], [[1.0 / 3], [1.0 / 3], [1.0 / 3]], mode="same", boundary="symm")
    vuv = data[:, 41]
    lf0 = np.sum(data[:, 42:51], axis=1) / 9
    lf0[vuv < 0.5] = inf_float
    write_binary_file(mgc, f_mgc)
    write_binary_file(lf0, f_lf0)

def mlpg_splitcmp(f_cmp, f_mgc, f_lf0):
    with open(f_cmp, 'rb') as fid:
        data = np.fromfile(fid, dtype=np.float32)
        data = np.reshape(data, [-1, 43])
    mgc = data[:, 0:41]
    vuv = data[:, 41]
    lf0 = data[:, 42]
    write_binary_file(mgc, f_mgc)
    write_binary_file(lf0, f_lf0)

def main(args):
    if os.path.exists(args.wav_dir):
        shutil.rmtree(args.wav_dir)
    os.mkdir(args.wav_dir)
    if os.path.exists(args.mgc_dir):
        shutil.rmtree(args.mgc_dir)
    os.mkdir(args.mgc_dir)
    if os.path.exists(args.lf0_dir):
        shutil.rmtree(args.lf0_dir)
    os.mkdir(args.lf0_dir)
    for filename in os.listdir(args.cmp_dir):
        name = filename.strip().split(".")[0]
        f_cmp = os.path.join(args.cmp_dir, "{}.cmp".format(name))
        f_mgc = os.path.join(args.mgc_dir, "{}.mgc".format(name))
        f_lf0 = os.path.join(args.lf0_dir, "{}.lf0".format(name))
        if args.domlpg:
            mlpg_splitcmp(f_cmp,f_mgc,f_lf0)
        else:
            pz_spiltcmp(f_cmp, f_mgc, f_lf0)
        if args.vocoder == "bd":
            os.system("./bd_vocoder {} {} {}".format(f_mgc, f_lf0, args.wav_dir))
            os.system("rm {}.pcm".format(os.path.join(args.wav_dir,name)))
        elif args.vocoder == "lsp":
            f_wav = os.path.join(args.wav_dir, "{}.wav".format(name))
            vocoder = os.path.join(sys.path[0],"lsp_vocoder_main")
            os.system("{} --gain 2 --lsp_file {} --pitch_file {} --wav_file {}".format(vocoder, f_mgc, f_lf0, f_wav))
        else:
            print("error:unknown uocoder name...")

def cmvn(cmvn_path, cmp_path):
    with open(cmp_path) as fid:
        orign_cmp = np.fromfile(fid)
        orign_cmp = np.reshape(orign_cmp, [-1,51])
    cmvn = np.load(cmvn_path)
    print(cmvn)
    norm_cmp = (orign_cmp - cmvn["mean_labels"]) / cmvn["stddev_labels"]
    denorm_cmp = norm_cmp * cmvn["stddev_labels"] + cmvn["mean_labels"]
    write_binary_file(norm_cmp, os.path.join("cmvn_test","norm_cmp.cmp"))
    write_binary_file(denorm_cmp, os.path.join("cmvn_test","denorm_cmp.cmp"))
    write_binary_file(orign_cmp, os.path.join("cmvn_test","orign_cmp.cmp"))

def win2linux(lf0_dir, mgc_dir, wav_dir):
    for item in os.listdir(lf0_dir):
        name, _ = os.path.splitext(item)
        f_lf0 = os.path.join(lf0_dir, "{}.lf0".format(name))
        f_mgc = os.path.join(mgc_dir, "{}.lsp".format(name))
        print(f_mgc)
        if not os.path.exists(wav_dir):
            os.mkdir(wav_dir)
        if os.path.exists(f_mgc):
        # convert windows feature to linux feature.
            win_mgc = np.fromfile(f_mgc, dtype=np.float32)
            win_mgc = np.reshape(win_mgc, (-1,41))
            np.savetxt("{}.b".format(os.path.join(mgc_dir,name)), win_mgc)
            lin_mgc = np.zeros((win_mgc.shape[0], 41), dtype = np.float32)
            lin_mgc[:,0] = win_mgc[:,40] - 1.7
            for i in range(0,40):
                lin_mgc[:,i+1] = win_mgc[:,i] * (2 * 3.1415926535898)
            lin_mgc_name = os.path.join(mgc_dir, name)
            np.savetxt("{}.txt".format(os.path.join(mgc_dir,name)), lin_mgc)
            write_binary_file(lin_mgc, lin_mgc_name)
            if args.vocoder == "bd":
                os.system("./bd_vocoder {} {} {}".format(f_mgc, f_lf0, wav_dir))
                os.system("rm {}.pcm".format(os.path.join(wav_dir,name)))
            elif args.vocoder == "lsp":
                f_wav = os.path.join(wav_dir, "{}.wav".format(name))
                # os.system("./lsp_vocoder_main  --gain 2.0 --lsp_file {} --pitch_file {} --wav_file {}".format(lin_mgc_name, f_lf0, f_wav))
                os.system("./lsp_vocoder_main  --gain 2.0 --lsp_file {} --pitch_file {} --wav_file {}".format(f_mgc, f_lf0, f_wav))
            else:
                print("error:unknown uocoder name...")

def genwav_split(lf0_dir, mgc_dir, wav_dir):
    for item in os.listdir(lf0_dir):
        name, _ = os.path.splitext(item)
        f_lf0 = os.path.join(lf0_dir, "{}.lf0".format(name))
        f_mgc = os.path.join(mgc_dir, "{}.mgc".format(name))
        if not os.path.exists(wav_dir):
            os.mkdir(wav_dir)
        if os.path.exists(f_mgc):
            if args.vocoder == "bd":
                os.system("./bd_vocoder {} {} {}".format(f_mgc, f_lf0, wav_dir))
                os.system("rm {}.pcm".format(os.path.join(wav_dir,name)))
            elif args.vocoder == "lsp":
                f_wav = os.path.join(wav_dir, "{}.wav".format(name))
                vocoder = os.path.join(sys.path[0],"lsp_vocoder_main")
                os.system("{} --gain 2 --lsp_file {} --pitch_file {} --wav_file {}".format(vocoder, f_mgc, f_lf0, f_wav))
            else:
                print("error:unknown uocoder name...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmp_dir", default="cmp")
    parser.add_argument("--wav_dir", default="wav")
    parser.add_argument("--mgc_dir", default="mgc")
    parser.add_argument("--lf0_dir", default="lf0")
    parser.add_argument("--vocoder", default="lsp")
    parser.add_argument("--domlpg",  default=False)
    parser.add_argument("--use_lf0_mgc", default=False)
    args = parser.parse_args()
    if args.use_lf0_mgc:
        genwav_split(args.lf0_dir, args.mgc_dir, args.wav_dir)
    else:
        main(args)
