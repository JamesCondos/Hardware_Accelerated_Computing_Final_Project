#include "srcnn.h"
#include <math.h>

//Total BRAM = 2.1 + 0.66 = 2.1

// Conv1: 9x9 kernel, SAME (replicate) padding
// Output-stationary and input stationary tiled convolution
// Tile size: 51x51 (fits evenly into 255x255 image)
void conv3(ftmap_t layer2_output_tile[N2][TILE_H][TILE_W],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t layer3_output_tile[N3][TILE_H][TILE_W])
{
#pragma HLS inline off
    const int P  = F3 / 2;   // input padding where P = ceil(K/2) where K = 9 i.e kernal dimensions

    //for output tile H x W x N x 32b = (8 x 51 x 51 x 32)/10e6 = 0.66 Mbits < 5.1 Mbits DRAM

    //loop over output tile dimensions
    for (int tile_h = 0; tile_h < TILE_H; tile_h++){
        for (int tile_w = 0; tile_w < TILE_W; tile_w++){
            layer3_output_tile[0][tile_h][tile_w] = conv3_biases[0];
        }
    }

   //N0 =1 so we dont even need a loop for our input features this time
   //so basically we dont even need an input feature tile
   //Do convolution alongside padding in this section
   for (int input_feat = 0; input_feat < N2; input_feat++){
       for (int th = 0; th < TILE_H; th++){
           for (int tw = 0; tw < TILE_W; tw++){
               int h = th;
               int w = tw;

               // do padding and then do MAC
               //iterate over kernel height and width
               for (int kh = 0; kh < F3; kh++){
                   for (int kw = 0; kw < F3; kw++){

                       //do same padding with edge extension
                       int pad_h = (int)fminf(fmaxf(h + kh - P, 0),TILE_H - 1);
                       int pad_w = (int)fminf(fmaxf(w + kw - P, 0),TILE_W - 1);

                       //perform MAC
                       layer3_output_tile[0][th][tw] +=
                           conv3_weights[0][input_feat][kh][kw] *
                           layer2_output_tile[input_feat][pad_h][pad_w];
                   }
               }
           }
       }
   }
}
