#include "srcnn.h"

void srcnn(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t output_ftmap[N3][H][W])
{

#pragma HLS PIPELINE off
    // keep the final input and output feature maps in DRAM
    static ftmap_t input_tile[N0][TILE_H][TILE_W];
    static ftmap_t layer1_output_tile[N1][TILE_H][TILE_W];
    static ftmap_t layer2_output_tile[N2][TILE_H][TILE_W];
    static ftmap_t layer3_output_tile[N3][TILE_H][TILE_W];
    // intermediate feature maps also stored in DRAM

    // sequentially call the three convolutional layers
    for (int tile_h = 0; tile_h < H; tile_h += TILE_H) {
        for (int tile_w = 0; tile_w < W; tile_w += TILE_W) {

            // create the input tile
            input_tiler(input_ftmap, input_tile, tile_h, tile_w);

            // sequential SRCNN pipeline
            conv1_tile(input_tile, conv1_weights, conv1_biases, layer1_output_tile);
            conv2(layer1_output_tile, conv2_weights, conv2_biases, layer2_output_tile);
            conv3(layer2_output_tile, conv3_weights, conv3_biases, layer3_output_tile);

            // map output tiles to the final upscaled image
            reconstructor(output_ftmap, layer3_output_tile, tile_h, tile_w);
        }
    }
}


// helper function to create the 17x17 input tile going into the first layer
void input_tiler(ftmap_t input_ftmap[N0][H][W],
                 ftmap_t input_tile[N0][TILE_H][TILE_W],
                 int tile_h, int tile_w)
{
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
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            output_ftmap[0][tile_h + i][tile_w + j] = output_tile[0][i][j];
        }
    }
}
