#!/bin/bash

./local/run_tts.sh \ #run_tts_world_mlpg.sh, run_tts_straight_mlpg.sh
    --input_dim=297 \
    --output_dim=199 \
    --batch_size=8 \
    --learning_rate=0.001 \
    --rnn_cell=fused_lstm \
    --max_epochs=30 \
    --dnn_num_hidden=256 \
    --rnn_num_hidden=256 \
    --dnn_depth=3 \
    --rnn_depth=2 \
    --bidirectional=True
    # --resume_training
