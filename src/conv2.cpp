#include "srcnn.h"
#include "hls_stream.h"
#include <math.h>

// Conv2: 1x1 kernel, no padding required
// Output-stationary tiled convolution
void conv2(
    hls::stream<ftmap_t> &conv1_to_conv2,
    param_t conv2_weights[N2][N1][F2][F2],
    param_t conv2_biases[N2],
    hls::stream<ftmap_t> &conv2_to_conv3)
{
#pragma HLS INLINE off
#pragma HLS STREAM variable=conv2_to_conv3 depth=512


	 //ensures that MACs are done on DSPs due to LUT shortage
#pragma HLS BIND_OP op=mul impl=dsp
#pragma HLS BIND_OP op=add impl=dsp

    ftmap_t input_tile[N1][TILE_H][TILE_W];
    ftmap_t layer2_output_tile[N2][TILE_H][TILE_W];


	//array partitioning for doing parallel MACS
#pragma HLS ARRAY_PARTITION variable=conv2_weights cyclic factor=4 dim=2  //input feature dimension being partitioned
#pragma HLS ARRAY_PARTITION variable=conv2_biases complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_tile cyclic factor=4 dim=1

    // Read streamed data from conv1
    STREAM_READ_CONV2:
    for (int feat = 0; feat < N1; feat++) {
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                input_tile[feat][i][j] = conv1_to_conv2.read();
            }
        }
    }

    // Initialize with bias
    OUT_STATIONARY_CONV2:
    for (int out_feat = 0; out_feat < N2; out_feat++) {
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE II=1
                layer2_output_tile[out_feat][i][j] = conv2_biases[out_feat];
            }
        }
    }

    // 1x1 convolution
    out_feature_loop_CONV2:
    for (int feat = 0; feat < N2; feat++) {
    	in_feature_loop_conv2:
        for (int in_feat = 0; in_feat < N1; in_feat++) {
#pragma HLS UNROLL FACTOR=4 //unroll this loop to create multiple parallel MAC calculations via DSPS
          	tile_height_loop:
            for (int i = 0; i < TILE_H; i++) {
            	tile_width_loop:
#pragma HLS PIPELINE II=1 //pipeline over the height and width here so that the next
				//MAC calculation can begin before the previous one has started
                for (int j = 0; j < TILE_W; j++) {
                    layer2_output_tile[feat][i][j] +=
                        conv2_weights[feat][in_feat][0][0] *
                        input_tile[in_feat][i][j];
                }
            }
        }

        // ReLU activation
        RELU_CONV2:
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE II = 1
                if (layer2_output_tile[feat][i][j] < 0) {
                    layer2_output_tile[feat][i][j] = 0;
                }
            }
        }
    }

    // Stream out to conv3
    STREAM_OUT_CONV2:
    for (int feat = 0; feat < N2; feat++) {
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                conv2_to_conv3.write(layer2_output_tile[feat][i][j]);
            }
        }
    }
}
