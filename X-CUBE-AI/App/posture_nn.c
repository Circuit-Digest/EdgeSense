/**
  ******************************************************************************
  * @file    posture_nn.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-09T15:33:31+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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


#include "posture_nn.h"
#include "posture_nn_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_posture_nn
 
#undef AI_POSTURE_NN_MODEL_SIGNATURE
#define AI_POSTURE_NN_MODEL_SIGNATURE     "0xc348ea0249be9da919b60fbe5f193eed"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-09-09T15:33:31+0530"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_POSTURE_NN_N_BATCHES
#define AI_POSTURE_NN_N_BATCHES         (1)

static ai_ptr g_posture_nn_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_posture_nn_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_posture_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_posture_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _fc1_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _relu2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  output_logits_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _fc1_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _fc1_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  output_logits_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  output_logits_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _conv_Conv_output_0_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &_Flatten_output_0_to_chlast_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_output0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 72, 1, 1), AI_STRIDE_INIT(4, 4, 4, 288, 288),
  1, &_Flatten_output_0_to_chlast_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _conv_Conv_output_0_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_conv_Conv_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _conv_Conv_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 3, 3), AI_STRIDE_INIT(4, 4, 4, 32, 96),
  1, &_conv_Conv_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _conv_Conv_output_0_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _conv_Conv_output_0_scratch1, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 6, 2), AI_STRIDE_INIT(4, 4, 4, 32, 192),
  1, &_conv_Conv_output_0_scratch1_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _conv_Conv_output_0_weights, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &_conv_Conv_output_0_weights_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _fc1_Gemm_output_0_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_fc1_Gemm_output_0_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _fc1_Gemm_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_fc1_Gemm_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _fc1_Gemm_output_0_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 72, 32, 1, 1), AI_STRIDE_INIT(4, 4, 288, 9216, 9216),
  1, &_fc1_Gemm_output_0_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _relu2_Relu_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_relu2_Relu_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  input_posture_Transpose_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 4, 4, 8, 64),
  1, &input_posture_Transpose_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  input_posture_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &input_posture_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  output_logits_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &output_logits_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  output_logits_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &output_logits_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  output_logits_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 32, 6, 1, 1), AI_STRIDE_INIT(4, 4, 128, 768, 768),
  1, &output_logits_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_logits_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_logits_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &output_logits_weights, &output_logits_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_logits_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &output_logits_chain,
  NULL, &output_logits_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _relu2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _relu2_Relu_output_0_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_relu2_Relu_output_0_chain,
  NULL, &output_logits_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc1_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Flatten_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc1_Gemm_output_0_weights, &_fc1_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc1_Gemm_output_0_layer, 5,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_fc1_Gemm_output_0_chain,
  NULL, &_relu2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Flatten_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_layer, 4,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Flatten_output_0_to_chlast_chain,
  NULL, &_fc1_Gemm_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_posture_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv_Conv_output_0_weights, &_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv_Conv_output_0_scratch0, &_conv_Conv_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _conv_Conv_output_0_layer, 3,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &_conv_Conv_output_0_chain,
  NULL, &_Flatten_output_0_to_chlast_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_posture_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_posture_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_posture_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_posture_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_posture_Transpose_chain,
  NULL, &_conv_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 10744, 1, 1),
    10744, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1096, 1, 1),
    1096, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_POSTURE_NN_IN_NUM, &input_posture_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_POSTURE_NN_OUT_NUM, &output_logits_output),
  &input_posture_Transpose_layer, 0x9df64376, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 10744, 1, 1),
      10744, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1096, 1, 1),
      1096, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_POSTURE_NN_IN_NUM, &input_posture_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_POSTURE_NN_OUT_NUM, &output_logits_output),
  &input_posture_Transpose_layer, 0x9df64376, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool posture_nn_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_posture_nn_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_posture_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 0);
    input_posture_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 0);
    input_posture_Transpose_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 512);
    input_posture_Transpose_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 512);
    _conv_Conv_output_0_scratch0_array.data = AI_PTR(g_posture_nn_activations_map[0] + 1024);
    _conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 1024);
    _conv_Conv_output_0_scratch1_array.data = AI_PTR(g_posture_nn_activations_map[0] + 0);
    _conv_Conv_output_0_scratch1_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 0);
    _conv_Conv_output_0_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 384);
    _conv_Conv_output_0_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 384);
    _Flatten_output_0_to_chlast_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 0);
    _Flatten_output_0_to_chlast_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 0);
    _fc1_Gemm_output_0_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 288);
    _fc1_Gemm_output_0_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 288);
    _relu2_Relu_output_0_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 0);
    _relu2_Relu_output_0_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 0);
    output_logits_output_array.data = AI_PTR(g_posture_nn_activations_map[0] + 128);
    output_logits_output_array.data_start = AI_PTR(g_posture_nn_activations_map[0] + 128);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool posture_nn_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_posture_nn_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv_Conv_output_0_weights_array.data = AI_PTR(g_posture_nn_weights_map[0] + 0);
    _conv_Conv_output_0_weights_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 0);
    _conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv_Conv_output_0_bias_array.data = AI_PTR(g_posture_nn_weights_map[0] + 576);
    _conv_Conv_output_0_bias_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 576);
    _fc1_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _fc1_Gemm_output_0_weights_array.data = AI_PTR(g_posture_nn_weights_map[0] + 608);
    _fc1_Gemm_output_0_weights_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 608);
    _fc1_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _fc1_Gemm_output_0_bias_array.data = AI_PTR(g_posture_nn_weights_map[0] + 9824);
    _fc1_Gemm_output_0_bias_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 9824);
    output_logits_weights_array.format |= AI_FMT_FLAG_CONST;
    output_logits_weights_array.data = AI_PTR(g_posture_nn_weights_map[0] + 9952);
    output_logits_weights_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 9952);
    output_logits_bias_array.format |= AI_FMT_FLAG_CONST;
    output_logits_bias_array.data = AI_PTR(g_posture_nn_weights_map[0] + 10720);
    output_logits_bias_array.data_start = AI_PTR(g_posture_nn_weights_map[0] + 10720);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_posture_nn_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_POSTURE_NN_MODEL_NAME,
      .model_signature   = AI_POSTURE_NN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 8434,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x9df64376,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_posture_nn_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_POSTURE_NN_MODEL_NAME,
      .model_signature   = AI_POSTURE_NN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 8434,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x9df64376,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_posture_nn_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_posture_nn_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_posture_nn_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_posture_nn_create(network, AI_POSTURE_NN_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_posture_nn_data_params_get(&params) != true) {
    err = ai_posture_nn_get_error(*network);
    return err;
  }
#if defined(AI_POSTURE_NN_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_POSTURE_NN_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_posture_nn_init(*network, &params) != true) {
    err = ai_posture_nn_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_posture_nn_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_posture_nn_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_posture_nn_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_posture_nn_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= posture_nn_configure_weights(net_ctx, params);
  ok &= posture_nn_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_posture_nn_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_posture_nn_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_POSTURE_NN_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

