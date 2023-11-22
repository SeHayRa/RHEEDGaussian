#include <iostream>

#include "rheed.h"
#include "parameters.h"

void rheed(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer15_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer15_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 150>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 6>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 2400>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 6912>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 48>(b11, "b11.txt");
        nnet::load_weights_from_txt<model_default_t, 1152>(w13, "w13.txt");
        nnet::load_weights_from_txt<model_default_t, 24>(b13, "b13.txt");
        nnet::load_weights_from_txt<model_default_t, 120>(w15, "w15.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b15, "b15.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1100
    nnet::conv_2d_cl<input_t, layer2_t, config2>(x, layer2_out, w2, b2); // layer1_0

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1100
    nnet::relu<layer2_t, layer4_t, ReLU_config4>(layer2_out, layer4_out); // layer1_2

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=66
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // layer1_3

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=14
    nnet::conv_2d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // layer2_0

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=14
    nnet::relu<layer6_t, layer8_t, ReLU_config8>(layer6_out, layer8_out); // layer2_2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=3
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // layer2_3

    auto& layer10_out = layer9_out;
    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::dense<layer9_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // fc

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::relu<layer11_t, layer12_t, ReLU_config12>(layer11_out, layer12_out); // relu

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::dense<layer12_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // fc1

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::relu<layer13_t, layer14_t, ReLU_config14>(layer13_out, layer14_out); // relu1

    nnet::dense<layer14_t, result_t, config15>(layer14_out, layer15_out, w15, b15); // fc2

}
