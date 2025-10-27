#ifndef _SRCNN_H_
#define _SRCNN_H_

#include "hls_stream.h"

//Tile dimensions for conv1 and conv2
#define TILE_H 17	//Tile Height
#define TILE_W 17	//Tile Width

// image dimensions
#define W  255          // image width
#define H  255          // image height
#define UP 3            // upscaling factor

// CNN dimensions
#define N0 1            // input features (image channels)
#define N1 64           // conv1 output features
#define F1 9            // conv1 kernel size
#define N2 32           // conv2 output features
#define F2 1            // conv2 kernel size
#define N3 1            // conv3 output features
#define F3 5            // conv3 kernel size

// data types
typedef float ftmap_t;  // feature map
typedef float param_t;  // parameters

// implements end-to-end SRCNN
void srcnn(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t output_ftmap[N3][H][W]);


void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W]);


// implements first convolutional layer of SRCNN (streaming output)
void conv1_tile(ftmap_t input_tile[N0][TILE_H][TILE_W],
                param_t conv1_weights[N1][N0][F1][F1],
                param_t conv1_biases[N1],
                hls::stream<ftmap_t> &conv1_to_conv2);

// implements second convolutional layer of SRCNN (streaming input/output)
void conv2(hls::stream<ftmap_t> &conv1_to_conv2,
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           hls::stream<ftmap_t> &conv2_to_conv3);

// implements third convolutional layer of SRCNN (streaming input)
void conv3(hls::stream<ftmap_t> &conv2_to_conv3,
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t layer3_output_tile[N3][TILE_H][TILE_W]);

void input_tiler(ftmap_t input_ftmap[N0][H][W],
                 ftmap_t input_tile[N0][TILE_H][TILE_W],
                 int tile_h, int tile_w);

void reconstructor(ftmap_t output_ftmap[N3][H][W],
                   ftmap_t output_tile[N3][TILE_H][TILE_W],
                   int tile_h, int tile_w);

#endif /* _SRCNN_H_ */
