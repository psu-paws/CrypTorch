// Author: Neha Jawalkar
// Modified by Kiwan Maeng for CrypTorch project.
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#ifndef __CONV2DINT__
#define __CONV2DINT__

struct Conv2DParams
{
    int bin, bout, N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight,
        zPadWLeft, zPadWRight,
        strideH, strideW, OH, OW;
    size_t size_I, size_F, size_O;
};

template <typename T>
struct GPUConv2DKey
{
    Conv2DParams p;
    size_t mem_size_I, mem_size_F, mem_size_O;
    T *I, *F, *O;
};

//template <typename T>
//void cutlass_conv2d(GPUConv2DKey<T> k, T *d_I, T *d_F, T *d_C);
void cutlass_conv2d64(Conv2DParams const &p, int64_t *d_I, int64_t *d_F, int64_t *d_C);
void cutlass_conv2d32(Conv2DParams const &p, int32_t *d_I, int32_t *d_F, int32_t *d_C);
// void cutlass_conv2d16(Conv2DParams const &p, int16_t *d_I, int16_t *d_F, int16_t *d_C);

#endif
