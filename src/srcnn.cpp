#include "srcnn.h"
#include "hls_stream.h"

void srcnn(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    param_t conv2_weights[N2][N1][F2][F2],
    param_t conv2_biases[N2],
    param_t conv3_weights[N3][N2][F3][F3],
    param_t conv3_biases[N3],
    ftmap_t output_ftmap[N3][H][W])
{
    // Execute the dataflow pipeline
#pragma HLS DATAFLOW
    // Loop over image tiles
	tile_height_loop_MAIN:
    for (int base_h = 0; base_h < H; base_h += TILE_H) {
    	tile_width_loop_MAIN:
        for (int base_w = 0; base_w < W; base_w += TILE_W) {

            // Local buffers for this tile
            ftmap_t input_tile[N0][TILE_H][TILE_W];
            ftmap_t layer3_output_tile[N3][TILE_H][TILE_W];
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=layer3_output_tile complete dim=1

            // Create FIFO streams between layers
            hls::stream<ftmap_t> conv1_to_conv2("conv1_to_conv2");
            hls::stream<ftmap_t> conv2_to_conv3("conv2_to_conv3");
#pragma HLS STREAM variable=conv1_to_conv2 depth=512
#pragma HLS STREAM variable=conv2_to_conv3 depth=512


            input_tiler(input_ftmap, input_tile, base_h, base_w);
            conv1_tile(input_tile, conv1_weights, conv1_biases, conv1_to_conv2);
            conv2(conv1_to_conv2, conv2_weights, conv2_biases, conv2_to_conv3);
            conv3(conv2_to_conv3, conv3_weights, conv3_biases, layer3_output_tile);

            // Write the reconstructed tile back to the full output map
            reconstructor(output_ftmap, layer3_output_tile, base_h, base_w);
        }
    }
}

// helper function to create the 17x17 input tile going into the first layer
void input_tiler(ftmap_t input_ftmap[N0][H][W],
                 ftmap_t input_tile[N0][TILE_H][TILE_W],
                 int tile_h, int tile_w)
{
#pragma HLS PIPELINE off
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            input_tile[0][i][j] = input_ftmap[0][tile_h + i][tile_w + j];
        }
    }
}


// helper function that maps the tile output from Conv3 to the full output feature map
void reconstructor(ftmap_t output_ftmap[N3][H][W],
                   ftmap_t output_tile[N3][TILE_H][TILE_W],
                   int tile_h, int tile_w)
{
#pragma HLS PIPELINE off
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            output_ftmap[0][tile_h + i][tile_w + j] = output_tile[0][i][j];
        }
    }
}
