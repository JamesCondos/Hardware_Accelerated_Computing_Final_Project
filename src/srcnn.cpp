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
    // Implement end-to-end SRCNN here
	static ftmap_t layer1_output[N1][H][W];
	static ftmap_t layer2_output[N2][H][W];
	conv1(input_ftmap, conv1_weights, conv1_biases, layer1_output);
	conv2(layer1_output, conv2_weights, conv2_biases, layer2_output);
	conv3(layer2_output, conv3_weights, conv3_biases, output_ftmap);
}
