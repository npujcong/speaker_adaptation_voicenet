#!/bin/bash
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright ASLP@NPU. All Rights Reserved
#
# Licensed under the Apache License, Veresion 2.0(the "License");
# You may not use the file except in compliance with the Licese.
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author npujcong@gmail.com(congjian)
# Date 2019/04/08 20:50:23
#
######################################################################

# Output features directory

out_dir=$1

cmp_dir="${out_dir}/cmp"
wav_dir="${out_dir}/wav"
lf0_dir="${out_dir}/lf0"
mgc_dir="${out_dir}/mgc"

python gain_wav.py \
    --wav_dir=${wav_dir} \
    --mgc_dir=${mgc_dir} \
    --lf0_dir=${lf0_dir} \
    --vocoder=lsp \
    --use_lf0_mgc=True \
    --domlpg=True
