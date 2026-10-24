/**
  ******************************************************************************
  * @file    surface_nn.h
  * @date    2026-10-24T15:26:49+0530
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_SURFACE_NN_DETAILS_H
#define STAI_SURFACE_NN_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_surface_nn_details = {
  .tensors = (const stai_tensor[10]) {
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 16, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_canonical_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 16, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_conv1_Conv_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 16, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu1_Relu_output_0_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 8, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_pool1_MaxPool_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 8, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_conv2_Conv_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 8, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu2_Relu_output_0_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 4, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_pool2_MaxPool_output_0_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_fc1_Gemm_output_0_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu3_Relu_output_0_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_logits_output" }
  },
  .nodes = (const stai_node_details[9]){
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _conv1_Conv_output_0 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _relu1_Relu_output_0 */
    {.id = 3, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _pool1_MaxPool_output_0 */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _conv2_Conv_output_0 */
    {.id = 5, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _relu2_Relu_output_0 */
    {.id = 6, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _pool2_MaxPool_output_0 */
    {.id = 8, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _fc1_Gemm_output_0 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* _relu3_Relu_output_0 */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} } /* output_logits */
  },
  .n_nodes = 9
};
#endif