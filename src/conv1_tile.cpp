#include "srcnn.h"
#include "hls_stream.h"
#include <math.h>

// Conv1: 9x9 kernel, SAME (replicate) padding
// Output-stationary tiled convolution
void conv1_tile(
    ftmap_t input_tile[N0][TILE_H][TILE_W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    hls::stream<ftmap_t> &conv1_to_conv2)
{
#pragma HLS INLINE off



#pragma HLS STREAM variable=conv1_to_conv2 depth=512


	// array partitioning to allow multiple reads of the arrays
	//this makes pipelining achieve II = 1
	//complete input tile partitioning required
#pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=3
#pragma HLS ARRAY_PARTITION variable=conv1_biases complete dim=1


    const int P = F1 / 2;


    ftmap_t layer1_output_tile[N1][TILE_H][TILE_W];

#pragma HLS ARRAY_PARTITION variable=layer1_output_tile complete dim=1


    // Initialize with bias
    OUT_STATIONARY_CONV1:
    for (int out_feat = 0; out_feat < N1; out_feat++) {
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE II =1
                layer1_output_tile[out_feat][i][j] = conv1_biases[out_feat];
            }
        }
    }

    // Convolution with SAME padding
    feature_loop_conv1:
    for (int feat = 0; feat < N1; feat++) {
    	tile_height_loop_conv1:
        for (int th = 0; th < TILE_H; th++) {
        	tile_width_loop_conv1:
            for (int tw = 0; tw < TILE_W; tw++) {
#pragma HLS PIPELINE II=1
            	kernel_height_loop_conv1:
                for (int kh = 0; kh < F1; kh++) {
                	kernel_width_loop_conv1:
                    for (int kw = 0; kw < F1; kw++) {
#pragma HLS UNROLL FACTOR=4
                        int pad_h = (int)fminf(fmaxf(th + kh - P, 0), TILE_H - 1);
                        int pad_w = (int)fminf(fmaxf(tw + kw - P, 0), TILE_W - 1);
                        layer1_output_tile[feat][th][tw] +=
                            conv1_weights[feat][0][kh][kw] *
                            input_tile[0][pad_h][pad_w];
                    }
                }
                // ReLU
                if (layer1_output_tile[feat][th][tw] < 0) {
                    layer1_output_tile[feat][th][tw] = 0;
                }
            }
        }
    }

    // Stream out tile data (N1 x TILE_H x TILE_W)
    STREAM_OUT_CONV1:
    for (int feat = 0; feat < N1; feat++) {
        for (int th = 0; th < TILE_H; th++) {
            for (int tw = 0; tw < TILE_W; tw++) {
                conv1_to_conv2.write(layer1_output_tile[feat][th][tw]);
            }
        }
    }
}
