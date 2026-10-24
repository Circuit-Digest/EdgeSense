/**
  ******************************************************************************
  * @file    posture_nn.h
  * @date    2026-10-24T15:26:59+0530
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
#ifndef STAI_POSTURE_NN_DETAILS_H
#define STAI_POSTURE_NN_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_posture_nn_details = {
  .tensors = (const stai_tensor[9]) {
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 2, 8, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_posture_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 8, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_posture_Transpose_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 6, 6, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_conv_Conv_output_0_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 6, 6, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu1_Relu_output_0_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 3, 3, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_pool_MaxPool_output_0_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 3, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Flatten_output_0_to_chlast_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_fc1_Gemm_output_0_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu2_Relu_output_0_output" },
   { .size_bytes = 24, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 6}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_logits_output" }
  },
  .nodes = (const stai_node_details[8]){
    {.id = 2, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* input_posture_Transpose */
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _conv_Conv_output_0 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _relu1_Relu_output_0 */
    {.id = 3, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _pool_MaxPool_output_0 */
    {.id = 4, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _Flatten_output_0_to_chlast */
    {.id = 5, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _fc1_Gemm_output_0 */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _relu2_Relu_output_0 */
    {.id = 7, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} } /* output_logits */
  },
  .n_nodes = 8
};
#endif