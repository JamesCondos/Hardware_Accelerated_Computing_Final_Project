#include "srcnn.h"
#include <math.h>

//Total BRAM = 2.1 + 0.66 = 2.1

// Conv2: 1x1 kernel (no spatial padding required)
// Output-stationary tiled convolution
// Tile size: 51x51 (fits evenly into 255x255 image)
void conv2(ftmap_t layer1_output_tile[N1][TILE_H][TILE_W],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           ftmap_t layer2_output_tile[N2][TILE_H][TILE_W])
{
#pragma HLS INLINE off
    // F2 = 1, so no padding needed
    const int P = 0;

    // for output tile H x W x N x 32b = (8 x 51 x 51 x 32)/10e6 = 0.66 Mbits < 5.1 Mbits DRAM

    // loop over output tile dimensions and initialize with bias
    for (int out_feat = 0; out_feat < N2; out_feat++) {
        for (int tile_h = 0; tile_h < TILE_H; tile_h++) {
            for (int tile_w = 0; tile_w < TILE_W; tile_w++) {
                layer2_output_tile[out_feat][tile_h][tile_w] = conv2_biases[out_feat];
            }
        }
    }

    // Do 1x1 convolution: multiply corresponding pixels across all input feature maps
    for (int feat = 0; feat < N2; feat++) {
        for (int input_feat = 0; input_feat < N1; input_feat++) {
            for (int th = 0; th < TILE_H; th++) {
                for (int tw = 0; tw < TILE_W; tw++) {
                    layer2_output_tile[feat][th][tw] +=
                        conv2_weights[feat][input_feat][0][0] *
                        layer1_output_tile[input_feat][th][tw];
                }
            }
        }

        // activation function (ReLU)
        for (int th = 0; th < TILE_H; th++) {
            for (int tw = 0; tw < TILE_W; tw++) {
                if (layer2_output_tile[feat][th][tw] < 0) {
                    layer2_output_tile[feat][th][tw] = 0;
                }
            }
        }
    }
}
