#include <torch/extension.h>
#include <ATen/ATen.h>
#include "conv2d_int.h"
#include <iostream>

void conv2d64(torch::Tensor &I, torch::Tensor &F, torch::Tensor &O, int N, int CI, int H, int W, int CO, int FH, int FW, int stride, int padding)
{
    struct Conv2DParams p = {
        // TODO: What is bin and bout?
        .bin=0, .bout=0, .N=N, .H=H, .W=W, .CI=CI, .FH=FH, .FW=FW, .CO=CO,
        .zPadHLeft=padding, .zPadHRight=padding, .zPadWLeft=padding, .zPadWRight=padding,
        .strideH=stride, .strideW=stride};
    p.OH = ((p.H - p.FH + (p.zPadHLeft + p.zPadHRight)) / p.strideH) + 1;
    p.OW = ((p.W - p.FW + (p.zPadWLeft + p.zPadWRight)) / p.strideW) + 1;
    p.size_I = p.N * p.H * p.W * p.CI;
    p.size_F = p.CO * p.FH * p.FW * p.CI;
    p.size_O = p.N * p.OH * p.OW * p.CO;
    // struct GPUConv2DKey<int64_t> k = {
    //     .p = p,
    //     // TODO: I don't think the below are used.
    //     //.mem_size_I = 0, .mem_size_F = 0, .mem_size_O = 0,
    //     //.I = I.data_ptr<int64_t>(), .F = F.data_ptr<int64_t>(), .O = O.data_ptr<int64_t>()
    // };
    cutlass_conv2d64(p, I.data_ptr<int64_t>(), F.data_ptr<int64_t>(), O.data_ptr<int64_t>());
}

void conv2d32(torch::Tensor &I, torch::Tensor &F, torch::Tensor &O, int N, int CI, int H, int W, int CO, int FH, int FW, int stride, int padding)
{
    struct Conv2DParams p = {
        // TODO: What is bin and bout?
        .bin=0, .bout=0, .N=N, .H=H, .W=W, .CI=CI, .FH=FH, .FW=FW, .CO=CO,
        .zPadHLeft=padding, .zPadHRight=padding, .zPadWLeft=padding, .zPadWRight=padding,
        .strideH=stride, .strideW=stride};
    p.OH = ((p.H - p.FH + (p.zPadHLeft + p.zPadHRight)) / p.strideH) + 1;
    p.OW = ((p.W - p.FW + (p.zPadWLeft + p.zPadWRight)) / p.strideW) + 1;
    p.size_I = p.N * p.H * p.W * p.CI;
    p.size_F = p.CO * p.FH * p.FW * p.CI;
    p.size_O = p.N * p.OH * p.OW * p.CO;
    // struct GPUConv2DKey<int32_t> k = {
    //     .p = p,
    //     // TODO: I don't think the below are used.
    //     //.mem_size_I = 0, .mem_size_F = 0, .mem_size_O = 0,
    //     //.I = I.data_ptr<int64_t>(), .F = F.data_ptr<int64_t>(), .O = O.data_ptr<int64_t>()
    // };
    cutlass_conv2d32(p, I.data_ptr<int32_t>(), F.data_ptr<int32_t>(), O.data_ptr<int32_t>());
}

// Couldn't get this to work, It seems like Cutlass is hardcoded is SIMT mode to access memory
// in one element sized blocks, however the GPU hardware requires accesses to be a minimum of 4 bytes
// in size. Given that int16_t is only 2 bytes big, this compiles with errors.

// void conv2d16(torch::Tensor &I, torch::Tensor &F, torch::Tensor &O, int N, int CI, int H, int W, int CO, int FH, int FW, int stride, int padding)
// {
//     struct Conv2DParams p = {
//         // TODO: What is bin and bout?
//         .bin=0, .bout=0, .N=N, .H=H, .W=W, .CI=CI, .FH=FH, .FW=FW, .CO=CO,
//         .zPadHLeft=padding, .zPadHRight=padding, .zPadWLeft=padding, .zPadWRight=padding,
//         .strideH=stride, .strideW=stride};
//     p.OH = ((p.H - p.FH + (p.zPadHLeft + p.zPadHRight)) / p.strideH) + 1;
//     p.OW = ((p.W - p.FW + (p.zPadWLeft + p.zPadWRight)) / p.strideW) + 1;
//     p.size_I = p.N * p.H * p.W * p.CI;
//     p.size_F = p.CO * p.FH * p.FW * p.CI;
//     p.size_O = p.N * p.OH * p.OW * p.CO;
//     // struct GPUConv2DKey<int32_t> k = {
//     //     .p = p,
//     //     // TODO: I don't think the below are used.
//     //     //.mem_size_I = 0, .mem_size_F = 0, .mem_size_O = 0,
//     //     //.I = I.data_ptr<int64_t>(), .F = F.data_ptr<int64_t>(), .O = O.data_ptr<int64_t>()
//     // };
//     cutlass_conv2d32(p, I.data_ptr<int16_t>(), F.data_ptr<int16_t>(), O.data_ptr<int16_t>());
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d64", &conv2d64);
  m.def("conv2d32", &conv2d32);
//   m.def("conv2d16", &conv2d16);
}
