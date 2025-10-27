#include "srcnn.h"
#include "hls_stream.h"
#include <math.h>

// Conv3: 5x5 kernel, SAME (replicate) padding
// Output-stationary tiled convolution
void conv3(
    hls::stream<ftmap_t> &conv2_to_conv3,
    param_t conv3_weights[N3][N2][F3][F3],
    param_t conv3_biases[N3],
    ftmap_t layer3_output_tile[N3][TILE_H][TILE_W])
{

#pragma HLS INLINE off
#pragma HLS STREAM variable=conv2_to_conv3 depth=512


    const int P = F3 / 2;

    ftmap_t input_tile[N2][TILE_H][TILE_W];

    // Read input tile from stream
    for (int feat = 0; feat < N2; feat++) {
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                input_tile[feat][i][j] = conv2_to_conv3.read();
            }
        }
    }

    // Initialize output with bias:
    OUT_STATIONARY_CONV3:
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE II = 1
            layer3_output_tile[0][i][j] = conv3_biases[0];
        }
    }

    // Perform convolution
    MAC_CONV3:
    for (int in_feat = 0; in_feat < N2; in_feat++) {
        for (int th = 0; th < TILE_H; th++) {
            for (int tw = 0; tw < TILE_W; tw++) {
                for (int kh = 0; kh < F3; kh++) {
                    for (int kw = 0; kw < F3; kw++) {
                        int pad_h = (int)fminf(fmaxf(th + kh - P, 0), TILE_H - 1);
                        int pad_w = (int)fminf(fmaxf(tw + kw - P, 0), TILE_W - 1);
                        layer3_output_tile[0][th][tw] +=
                            conv3_weights[0][in_feat][kh][kw] *
                            input_tile[in_feat][pad_h][pad_w];
                    }
                }
            }
        }
    }
}
