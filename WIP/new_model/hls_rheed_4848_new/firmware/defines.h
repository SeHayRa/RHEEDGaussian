#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 48
#define N_INPUT_2_1 48
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 44
#define OUT_WIDTH_2 44
#define N_FILT_2 6
#define OUT_HEIGHT_2 44
#define OUT_WIDTH_2 44
#define N_FILT_2 6
#define OUT_HEIGHT_5 11
#define OUT_WIDTH_5 11
#define N_FILT_5 6
#define OUT_HEIGHT_6 7
#define OUT_WIDTH_6 7
#define N_FILT_6 16
#define OUT_HEIGHT_6 7
#define OUT_WIDTH_6 7
#define N_FILT_6 16
#define OUT_HEIGHT_9 3
#define OUT_WIDTH_9 3
#define N_FILT_9 16
#define N_SIZE_0_10 144
#define N_LAYER_11 48
#define N_LAYER_11 48
#define N_LAYER_13 24
#define N_LAYER_13 24
#define N_LAYER_15 5

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 6*1> layer2_t;
typedef nnet::array<ap_fixed<16,6>, 6*1> layer4_t;
typedef ap_fixed<18,8> layer1_2_table_t;
typedef nnet::array<ap_fixed<16,6>, 6*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer6_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer8_t;
typedef ap_fixed<18,8> layer2_2_table_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 48*1> layer11_t;
typedef ap_uint<1> layer11_index;
typedef nnet::array<ap_fixed<16,6>, 48*1> layer12_t;
typedef ap_fixed<18,8> relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 24*1> layer13_t;
typedef ap_uint<1> layer13_index;
typedef nnet::array<ap_fixed<16,6>, 24*1> layer14_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;
typedef ap_uint<1> layer15_index;

#endif
