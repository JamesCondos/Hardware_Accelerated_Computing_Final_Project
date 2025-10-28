#include "srcnn.h"
#include <math.h>

//Total BRAM = 2.1 + 0.66 = 2.1

// Conv1: 9x9 kernel, SAME (replicate) padding
// Output-stationary and input stationary tiled convolution
// Tile size: 51x51 (fits evenly into 255x255 image)
void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W])
{
    const int P  = F1 / 2;   // input padding where P = ceil(K/2) where K = 9 i.e kernal dimensions

    //OUTPUT TILING SECTION
    const int out_feat_tile = TILE_N; // output feature maps per tile

    //set tiles to be 51x51 so that we can divide 255 into 5 tiles ,
    //this means we have a
    const int tile_height = TILE_H; // tile height
    const int tile_width = TILE_W;  // tile width

    //for output tile H x W x N x 32b = (8 x 51 x 51 x 32)/10e6 = 0.66 Mbits < 5.1 Mbits DRAM

    //loop over output tile dimensions
    for (int out_feat = 0; out_feat < N1; out_feat += out_feat_tile) {

        for (int out_height = 0; out_height < H; out_height += tile_height) {

            for (int out_width = 0; out_width < W; out_width += tile_width) {

                //create output tile variable
                ftmap_t output_tile[out_feat_tile][tile_height][tile_width];

                //create output tile in bram and intiialize with bias
                for (int of = 0; of < out_feat_tile && (out_feat + of) < N1; of++) {
                    for (int th = 0; th < tile_height; th++) {
                        for (int tw = 0; tw < tile_width; tw++) {
                            output_tile[of][th][tw] = conv1_biases[out_feat + of];
                        }
                    }
                }

                //N0 =1 so we dont even need a loop for our input features this time
                //so basically we dont even need an input feature tile
                for (int of = 0; of < out_feat_tile && (out_feat + of) < N1; of++) {
                    for (int th = 0; th < tile_height; th++) {
                        for (int tw = 0; tw < tile_width; tw++) {

                            // compute global coordinates of this output pixel
                            int global_y = out_height + th;
                            int global_x = out_width + tw;

                            // perform convolution
                            for (int kh = 0; kh < F1; ++kh) {
                                for (int kw = 0; kw < F1; ++kw) {

                                    // edge extension to the padding where padded parts have the same values as the closest pixel
                                    int input_y = (int)fminf(fmaxf(global_y + kh - P, 0), H - 1);
                                    int input_x = (int)fminf(fmaxf(global_x + kw - P, 0), W - 1);

                                    // read directly from global input feature map instead of local_input
                                    output_tile[of][th][tw] +=
                                        conv1_weights[out_feat + of][0][kh][kw] *
                                        input_ftmap[0][input_y][input_x];
                                }
                            }

                            //activation function
                            if (output_tile[of][th][tw] < 0) {
                                output_tile[of][th][tw] = 0;
                            }
                        }
                    }
                }

                // write output tile back into the main feature map
                for (int of = 0; of < out_feat_tile && (out_feat + of) < N1; ++of) {
                    for (int th = 0; th < tile_height; ++th) {
                        for (int tw = 0; tw < tile_width; ++tw) {
                            output_ftmap[out_feat + of][out_height + th][out_width + tw] =
                                output_tile[of][th][tw];
                        }
                    }
                }

            }

        }

    }

}
