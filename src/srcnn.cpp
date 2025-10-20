#include "srcnn.h"
#include <math.h>

// ================= Tunables =================
#ifndef Tile_Height
#define Tile_Height 16
#endif
#ifndef Tile_Width
#define Tile_Width  16
#endif

// Set to achieved float-adder II from your report if higher (e.g., 7–8)
#ifndef ACC_II
#define ACC_II 6
#endif

// Parallelism knobs
#ifndef UF_C1
#define UF_C1 4       // conv1 output-channel unroll (use 4 -> 2 -> 1 if ports/timing complain)
#endif
#ifndef UF_N2
#define UF_N2 1       // keep conv2 serial for now; raise to 2 later if clean
#endif

#define Padding_1  (F1/2)                        // 4 for 9x9
#define Padding_2  (F2/2)                        // 0 for 1x1
#define Padding_3  (F3/2)                        // 2 for 5x5
#define Padding_Total (Padding_1 + Padding_2 + Padding_3) // 6 for 9-1-5


static inline int clampi(int v, int lo, int hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// No-FMA mul+add barrier to mirror golden's "acc += w * x;"
static inline param_t mad_no_fma(param_t a, param_t b, param_t s) {
  volatile param_t p = a * b;  // rounded multiply
  return s + p;                // then rounded add
}

// -------------- load_patch_tile -------------
static void load_patch_tile(
    ftmap_t input_ftmap[N0][H][W],
    int h0, int w0, int th_eff, int tw_eff,
    ftmap_t patch[Tile_Height + 2*Padding_Total]
                 [Tile_Width  + 2*Padding_Total])
{
  const int PH = th_eff + 2*Padding_Total;
  const int PW = tw_eff + 2*Padding_Total;

#pragma HLS INLINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS DEPENDENCE variable=patch inter false
#pragma HLS DEPENDENCE variable=patch intra false

  for (int py = 0; py < PH; ++py) {
    for (int px = 0; px < PW; ++px) {
#pragma HLS PIPELINE
      const int yy = clampi(h0 + py - Padding_Total, 0, H-1);
      const int xx = clampi(w0 + px - Padding_Total, 0, W-1);
      patch[py][px] = input_ftmap[0][yy][xx]; // N0==1 typical
    }
  }
}

// ---- conv1: all c1 at a clamped center (banked + constant-bound unroll) ----
static void conv1_all_c1_at_clamped_center(
    ftmap_t patch[Tile_Height + 2*Padding_Total]
                 [Tile_Width  + 2*Padding_Total],
    int h0, int w0, int gyc, int gxc,
    param_t  conv1_w[N1][N0][F1][F1],
    param_t  conv1_b[N1],
    param_t  c1_vec[N1])
{
#pragma HLS INLINE
#pragma HLS DEPENDENCE variable=c1_vec inter false
#pragma HLS DEPENDENCE variable=c1_vec intra false

  // Bank weights/bias along N1 so UF_C1 lanes have their own ROM banks
#pragma HLS RESOURCE        variable=conv1_w core=ROM_2P
#pragma HLS ARRAY_PARTITION variable=conv1_w block factor=UF_C1 dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_b block factor=UF_C1 dim=1

  for (int c1b = 0; c1b < N1; c1b += UF_C1) {
    const int lanes = (c1b + UF_C1 <= N1) ? UF_C1 : (N1 - c1b);

    // Per-lane accumulators in registers
    param_t v_lane[UF_C1];
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1

    // Init with bias (guard for tail)
    for (int u = 0; u < UF_C1; ++u) {
#pragma HLS UNROLL
      v_lane[u] = (u < lanes) ? conv1_b[c1b + u] : 0.0f;
    }

    // 9x9 kernel MACs; broadcast x to all lanes; guard tail with (u<lanes)
    for (int ky = 0; ky < F1; ++ky) {
      const int gh = clampi(gyc + ky - Padding_1, 0, H-1);
      const int py = (gh - h0) + Padding_Total;
      for (int kx = 0; kx < F1; ++kx) {
#pragma HLS PIPELINE II=ACC_II
        const int gw = clampi(gxc + kx - Padding_1, 0, W-1);
        const int px = (gw - w0) + Padding_Total;
        const param_t x = patch[py][px];

        for (int u = 0; u < UF_C1; ++u) {
#pragma HLS UNROLL
          if (u < lanes) {
            v_lane[u] = mad_no_fma(conv1_w[c1b + u][0][ky][kx], x, v_lane[u]);
          }
        }
      }
    }

    // ReLU + write back
    for (int u = 0; u < UF_C1; ++u) {
#pragma HLS UNROLL
      if (u < lanes) c1_vec[c1b + u] = fmaxf(0.0f, v_lane[u]);
    }
  }
}

// ------------- conv2 (1x1) from c1 vector (weights cached) -------------------
static param_t conv2_single_from_c1(
    int n2,
    param_t  conv2_w[N2][N1][F2][F2],
    param_t  conv2_b[N2],
    const param_t c1_vec[N1])
{
#pragma HLS INLINE  // allow pipelining across pixel loop

  param_t wrow[N1];
#pragma HLS ARRAY_PARTITION variable=wrow complete dim=1
  for (int c1 = 0; c1 < N1; ++c1) {
#pragma HLS PIPELINE II=ACC_II
    wrow[c1] = conv2_w[n2][c1][0][0];
  }

  param_t acc = conv2_b[n2];
  for (int c1 = 0; c1 < N1; ++c1) {
#pragma HLS PIPELINE II=ACC_II
    acc = mad_no_fma(wrow[c1], c1_vec[c1], acc);
  }
  return fmaxf(0.0f, acc);
}

// --------------- precompute conv1 & conv2 on the conv3 halo ------------------
static void precompute_conv12_halo(
    ftmap_t  patch[Tile_Height + 2*Padding_Total]
                  [Tile_Width  + 2*Padding_Total],
    int h0, int w0, int th_eff, int tw_eff,
    param_t  conv1_w[N1][N0][F1][F1],
    param_t  conv1_b[N1],
    param_t  conv2_w[N2][N1][F2][F2],
    param_t  conv2_b[N2],
    ftmap_t  conv2_buf[N2]
                      [Tile_Height + 2*Padding_3]
                      [Tile_Width  + 2*Padding_3])
{
#pragma HLS INLINE off
#pragma HLS RESOURCE variable=conv1_w core=ROM_2P
#pragma HLS RESOURCE variable=conv2_w core=ROM_2P

  const int C2H = th_eff + 2*Padding_3;
  const int C2W = tw_eff + 2*Padding_3;

  param_t c1_vec[N1];
#pragma HLS DEPENDENCE variable=c1_vec inter false
#pragma HLS DEPENDENCE variable=c1_vec intra false

  for (int yi = 0; yi < C2H; ++yi) {
    const int gyc = clampi(h0 + yi - Padding_3, 0, H-1);
    for (int xi = 0; xi < C2W; ++xi) {
#pragma HLS PIPELINE II=1   // start a new (yi,xi) every cycle (II will be governed by conv1)
      const int gxc = clampi(w0 + xi - Padding_3, 0, W-1);

      // conv1 (all c1) at clamped center — parallel across UF_C1 lanes
      conv1_all_c1_at_clamped_center(patch, h0, w0, gyc, gxc,
                                     conv1_w, conv1_b, c1_vec);

      // conv2 for all n2 at this (yi,xi) — kept serial initially
      for (int n2 = 0; n2 < N2; ++n2) {
#if UF_N2 > 1
#pragma HLS UNROLL factor=UF_N2
#endif
        conv2_buf[n2][yi][xi] =
            conv2_single_from_c1(n2, conv2_w, conv2_b, c1_vec);
      }
    }
  }
}

// ------------------------- conv3 (row-register accumulator) ------------------
static void conv3_from_precomputed_conv2(
    int h0, int w0, int th_eff, int tw_eff,
    param_t  conv3_w[N3][N2][F3][F3],
    param_t  conv3_b[N3],
    ftmap_t  conv2_buf[N2]
                      [Tile_Height + 2*Padding_3]
                      [Tile_Width  + 2*Padding_3],
    ftmap_t  output_ftmap[N3][H][W])
{
#pragma HLS INLINE off
#pragma HLS RESOURCE variable=conv3_w core=ROM_2P

  const int C2H = th_eff + 2*Padding_3;
  const int C2W = tw_eff + 2*Padding_3;

  for (int o = 0; o < N3; ++o) {
    for (int oy = 0; oy < th_eff; ++oy) {

      param_t acc3_row[Tile_Width];
#pragma HLS ARRAY_PARTITION variable=acc3_row complete dim=1

      // init bias
      for (int ox = 0; ox < tw_eff; ++ox) {
#pragma HLS PIPELINE II=ACC_II
        acc3_row[ox] = conv3_b[o];
      }

      // accumulate over n2 and 5x5 taps
      for (int n2 = 0; n2 < N2; ++n2) {
        // cache kernel once per (o,n2)
        param_t k[F3][F3];
#pragma HLS ARRAY_PARTITION variable=k complete dim=0
        for (int ky = 0; ky < F3; ++ky)
          for (int kx = 0; kx < F3; ++kx) {
#pragma HLS PIPELINE II=ACC_II
            k[ky][kx] = conv3_w[o][n2][ky][kx];
          }

        for (int ky = 0; ky < F3; ++ky) {
          int gy = clampi(h0 + oy + ky - Padding_3, 0, H-1);
          int yi = (gy - h0) + Padding_3;
          if (yi < 0) yi = 0; else if (yi >= C2H) yi = C2H - 1;

          for (int kx = 0; kx < F3; ++kx) {
            for (int ox = 0; ox < tw_eff; ++ox) {
#pragma HLS PIPELINE II=ACC_II
              int gx = clampi(w0 + ox + kx - Padding_3, 0, W-1);
              int xi = (gx - w0) + Padding_3;
              if (xi < 0) xi = 0; else if (xi >= C2W) xi = C2W - 1;

              const ftmap_t c2 = conv2_buf[n2][yi][xi];
              acc3_row[ox] = mad_no_fma(k[ky][kx], c2, acc3_row[ox]);
            }
          }
        }
      }

      // write row
      for (int ox = 0; ox < tw_eff; ++ox) {
#pragma HLS PIPELINE II=ACC_II
        output_ftmap[o][h0 + oy][w0 + ox] = acc3_row[ox];
      }
    }
  }
}


void srcnn(ftmap_t input_ftmap[N0][H][W],
           param_t  conv1_weights[N1][N0][F1][F1],
           param_t  conv1_biases[N1],
           param_t  conv2_weights[N2][N1][F2][F2],
           param_t  conv2_biases[N2],
           param_t  conv3_weights[N3][N2][F3][F3],
           param_t  conv3_biases[N3],
           ftmap_t  output_ftmap[N3][H][W])
{
#pragma HLS INLINE off
#pragma HLS LOOP_FLATTEN off

  ftmap_t patch[Tile_Height + 2*Padding_Total]
                [Tile_Width  + 2*Padding_Total];

  ftmap_t conv2_buf[N2]
                   [Tile_Height + 2*Padding_3]
                   [Tile_Width  + 2*Padding_3];
#pragma HLS BIND_STORAGE variable=conv2_buf type=ram_2p impl=bram

  for (int h0 = 0; h0 < H; h0 += Tile_Height) {
    const int th_eff = (h0 + Tile_Height <= H) ? Tile_Height : (H - h0);

    for (int w0 = 0; w0 < W; w0 += Tile_Width) {
      const int tw_eff = (w0 + Tile_Width <= W) ? Tile_Width : (W - w0);

      load_patch_tile(input_ftmap, h0, w0, th_eff, tw_eff, patch);

      precompute_conv12_halo(
          patch, h0, w0, th_eff, tw_eff,
          conv1_weights, conv1_biases,
          conv2_weights, conv2_biases,
          conv2_buf);

      conv3_from_precomputed_conv2(
          h0, w0, th_eff, tw_eff,
          conv3_weights, conv3_biases,
          conv2_buf, output_ftmap);
    }
  }
}
