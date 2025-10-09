#include "srcnn.h"

// implements conv1 layer of SRCNN does fetch and extraction
void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1], //F1 = 9 therefore we have 9x9 filter kernel
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W])
{
    // implement conv1 layer of SRCNN here

	//outermost loop where we iterate over the output feature map position
	for (int n = 0 ; n < N1 ; n++){ //iterate over output feature maps
		//nested loops to iterate over the output image pixel dimensions
		for ( int h = 0; h < H; h++){ //output image height
			for ( int w = 0; w < W; w++){ //output image width


				//inner nexted loops to iterate over filter kernel dimensions and input feature map
				for ( int i = 0; i < F1; i++){ //iterate over the kernel height
					for (int j = 0;  j < F1 ; j++){ //iterate over kernel width
						for (int k = 0; k < N0; k++){ //iterate over input feature map

							//perform convolution of input image pixel areal with the kernel image and place
							//in the current output feature map position based on outermost loops
							output_ftmap[n][h][w] += conv1_weights[n][k][i][j] * input_ftmap[k][h+i][w+j];

							//for padding check if filter kenrel [k i j] is within  input ft map[k h w]
							//use if else, types of padding
							//zero padding means if u set the outpuit ftmap to 0 or do the convolution above
							//zero padding backup (inaccuracies)
							//preferred is interpolated padding
						}
					}
				}
				if outputfile <= 0
					outfile = 0
					else
					outfile = outfile +
			}
		}
	}
}
