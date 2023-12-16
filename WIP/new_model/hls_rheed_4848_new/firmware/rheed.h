#ifndef RHEED_H_
#define RHEED_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void rheed(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer15_out
);

#endif
