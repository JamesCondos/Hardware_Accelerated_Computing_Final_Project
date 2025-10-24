#include "srcnn.h"
#include <hls_stream.h>

void srcnn(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t output_ftmap[N3][H][W])
{

//#pragma HLS PIPELINE off
#pragma HLS DATAFLOW
    // keep the final input and output feature maps in DRAM
    hls::stream<ftmap_t> input_tile("input_tile");
    hls::stream<ftmap_t> layer1_output_tile("layer1_output_tile");
    hls::stream<ftmap_t> layer2_output_tile("layer2_output_tile");
    hls::stream<ftmap_t> layer3_output_tile("layer3_output_tile");
    hls::stream<int> tile_h1("tile_h1");
	hls::stream<int> tile_w1("tile_w1");
	hls::stream<int> tile_h2("tile_h2");
	hls::stream<int> tile_w2("tile_w2");
#pragma HLS STREAM variable=input_tile depth=16
#pragma HLS STREAM variable=layer1_output_tile depth=16
#pragma HLS STREAM variable=layer2_output_tile depth=16
#pragma HLS STREAM variable=layer3_output_tile depth=16
#pragma HLS STREAM variable=tile_h1 depth=32
#pragma HLS STREAM variable=tile_h2 depth=32
#pragma HLS STREAM variable=tile_w1 depth=32
#pragma HLS STREAM variable=tile_w2 depth=32
    // intermediate feature maps also stored in DRAM

    // sequentially call the three convolutional layers
//    for (int tile_h = 0; tile_h < H; tile_h += TILE_H) {
//        for (int tile_w = 0; tile_w < W; tile_w += TILE_W) {
    		tile_shifter(tile_h1, tile_w1, tile_h2, tile_w2);
            // create the input tile
            input_tiler(input_ftmap, input_tile, tile_h1, tile_w1);
            // sequential SRCNN pipeline
            conv1_tile(input_tile, conv1_weights, conv1_biases, layer1_output_tile);
            conv2(layer1_output_tile, conv2_weights, conv2_biases, layer2_output_tile);
            conv3(layer2_output_tile, conv3_weights, conv3_biases, layer3_output_tile);
            // map output tiles to the final upscaled image
            reconstructor(output_ftmap, layer3_output_tile, tile_h2, tile_w2);
//        }
//    }
}

void tile_shifter(hls::stream<int> &tile_h1, hls::stream<int> &tile_w1,
		hls::stream<int> &tile_h2, hls::stream<int> &tile_w2){
#pragma HLS INLINE off

//	int tile_h_read = tile_h.read();
//	int tile_w_read = tile_w.read();
	for (int tile_h = 0; tile_h < H; tile_h += TILE_H) {
	        for (int tile_w = 0; tile_w < W; tile_w += TILE_W) {
#pragma HLS PIPELINE
	        	tile_h1.write(tile_h);
				tile_h2.write(tile_h);
				tile_w1.write(tile_w);
				tile_w2.write(tile_w);
	        }
	}
}

// helper function to create the 17x17 input tile going into the first layer
void input_tiler(ftmap_t input_ftmap[N0][H][W],
				 hls::stream<ftmap_t> &input_tile,
				 hls::stream<int> &tile_h, hls::stream<int> &tile_w)
{
#pragma HLS INLINE off
	int tile_h_read = tile_h.read();
	int tile_w_read = tile_w.read();
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE
            input_tile.write(input_ftmap[0][tile_h_read + i][tile_w_read + j]);
        }
    }
}


// helper function that maps the tile output from Conv3 to the full output feature map
void reconstructor(ftmap_t output_ftmap[N3][H][W],
				   hls::stream<ftmap_t> &output_tile,
				   hls::stream<int> &tile_h, hls::stream<int> &tile_w)
{
#pragma HLS INLINE off
	int tile_h_read = tile_h.read();
	int tile_w_read = tile_w.read();
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
#pragma HLS PIPELINE
            output_ftmap[0][tile_h_read + i][tile_w_read + j] = output_tile.read();
        }
    }
}
