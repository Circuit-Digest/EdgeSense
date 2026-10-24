/**
  ******************************************************************************
  * @file    gesture_nn.h
  * @date    2026-10-24T15:26:39+0530
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
#ifndef STAI_GESTURE_NN_DETAILS_H
#define STAI_GESTURE_NN_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_gesture_nn_details = {
  .tensors = (const stai_tensor[12]) {
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 15, 8, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 8, 15}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_Transpose_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 8, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_conv1_Conv_output_0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 8, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu1_Relu_output_0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 4, 4, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_pool1_MaxPool_output_0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 4, 4, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_conv2_Conv_output_0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 4, 4, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu2_Relu_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 2, 2, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_pool2_AveragePool_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 32, 2, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_flatten_Flatten_output_0_to_chlast_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_fc1_Gemm_output_0_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_relu3_Relu_output_0_output" },
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_output" }
  },
  .nodes = (const stai_node_details[11]){
    {.id = 2, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* input_Transpose */
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _conv1_Conv_output_0 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _relu1_Relu_output_0 */
    {.id = 3, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _pool1_MaxPool_output_0 */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _conv2_Conv_output_0 */
    {.id = 5, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _relu2_Relu_output_0 */
    {.id = 6, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _pool2_AveragePool_output_0 */
    {.id = 7, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* _flatten_Flatten_output_0_to_chlast */
    {.id = 8, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* _fc1_Gemm_output_0 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* _relu3_Relu_output_0 */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} } /* output */
  },
  .n_nodes = 11
};
#endif