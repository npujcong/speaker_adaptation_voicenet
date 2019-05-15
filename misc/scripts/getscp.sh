#!/bin/bash

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
# Date 2019/01/26 10:58:46

dir1=$1
dir2=$2
out_scp=$3
ext=$4

if [ $ext -eq 0 ];then
comm -12 \
  <(ls $dir1 | awk -F . '{print $1}') \
  <(ls $dir2 | awk -F . '{print $1}') \
  > $out_scp
fi

if [ $ext -eq 1 ];then
comm -12 \
  <(ls $dir1) \
  <(ls $dir2) \
  > $out_scp
fi
