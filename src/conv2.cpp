#include "srcnn.h"
#include <math.h>

// implements conv2 layer of SRCNN
void conv2(ftmap_t input_ftmap[N1][H][W],
		   param_t conv2_weights[N2][N1][F2][F2], //F2 = 1 therefore we have 1x1 filter kernel. This is just a one dimensional convolution over a 2D input feature map
		   param_t conv2_biases[N2],
           ftmap_t output_ftmap[N2][H][W])
{
	//int padding = F2/2; //If we pick a coordinate on the top row of a 9 by 9 grid, we need at least 4 pixels surrounding it
    //no padding required here because this just uses a 1x1 filter kernal so gives us a 255x255 output as well

	for (int out_feat = 0; out_feat < N2; out_feat++) { //Loop over every output feature map

			for (int out_feat_y = 0; out_feat_y < H; out_feat_y++) { //Loop over the height of a feature map output

				for (int out_feat_x = 0; out_feat_x < W; out_feat_x++) { //Loop over the width of a feature map to capture a single pixel


					/*We have now picked a cell to give output to.
					 * We must now loop over a kernel and convolve to find the output for this cell
					 * Remember we must include the bias. each feature map has a single bias
					 */

				    float feat_bias = conv2_biases[out_feat];
				    float convolution = 0;

					for (int in_feat = 0; in_feat < N1 ; in_feat++) { //Loop over every input feature map (3 for RGB for example)
								//int new_ftmap_height = fmin(fmax(out_feat_y + kernel_y - padding, 0), H - 1);
								//int new_ftmap_width = fmin(fmax(out_feat_x + kernel_x - padding, 0), W - 1);
								convolution += conv2_weights[out_feat][in_feat][0][0]*input_ftmap[in_feat][out_feat_y][out_feat_x];
							}
					output_ftmap[out_feat][out_feat_y][out_feat_x] = fmaxf(0, convolution + feat_bias); //activation function (convolve could be negative)
				}
			}
		}
}
