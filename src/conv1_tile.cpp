
#include "srcnn.h"
#include <math.h>

//Total BRAM = 2.1 + 0.66 = 2.1

// Conv1: 9x9 kernel, SAME (replicate) padding
// Output-stationary and input stationary tiled convolution
// Tile size: 51x51 (fits evenly into 255x255 image)
void conv1_tile(ftmap_t input_tile[N0][TILE_H][TILE_W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t layer1_output_tile[N1][TILE_H][TILE_W])
{
#pragma HLS inline off
    const int P  = F1 / 2;   // input padding where P = ceil(K/2) where K = 9 i.e kernal dimensions

    //for output tile H x W x N x 32b = (8 x 51 x 51 x 32)/10e6 = 0.66 Mbits < 5.1 Mbits DRAM

    //loop over output tile dimensions
    for (int out_feat = 0; out_feat < N1; out_feat++){
    	for (int tile_h = 0; tile_h < TILE_H; tile_h++){
    		for (int tile_w = 0; tile_w < TILE_W; tile_w++){
    			layer1_output_tile[out_feat][tile_h][tile_w] = conv1_biases[out_feat];
    		}
    	}

    }

   //N0 =1 so we dont even need a loop for our input features this time
   //so basically we dont even need an input feature tile
   //Do convolution alongside padding in this section
   for (int feat = 0; feat < N1; feat++){
	   for (int th = 0; th < TILE_H; th++){
		   for (int tw = 0; tw < TILE_W; tw++){
			   int h = th;
			   int w = tw;

			   // do padding and then do MAC
			   //iterate over kernel height and width
			   for (int kh = 0; kh < F1; kh++){
				   for (int kw = 0; kw < F1; kw++){

					   //do same padding with edge extension
					   int pad_h = (int)fminf(fmaxf(h + kh - P, 0),TILE_H - 1);
					   int pad_w = (int)fminf(fmaxf(w + kw - P, 0), TILE_W - 1);

					   //perform MAC
					   layer1_output_tile[feat][th][tw] += conv1_weights[feat][0][kh][kw] * input_tile[0][pad_h][pad_w];

				   }
			   }
			  //activation functiion
			  if (layer1_output_tile[feat][th][tw]<0){
				  layer1_output_tile[feat][th][tw]= 0;
			  }
		   }
	   }
   }

}
