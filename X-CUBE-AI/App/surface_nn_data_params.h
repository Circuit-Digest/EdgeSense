/**
  ******************************************************************************
  * @file    surface_nn_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-09T16:02:45+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef SURFACE_NN_DATA_PARAMS_H
#define SURFACE_NN_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_SURFACE_NN_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_surface_nn_data_weights_params[1]))
*/

#define AI_SURFACE_NN_DATA_CONFIG               (NULL)


#define AI_SURFACE_NN_DATA_ACTIVATIONS_SIZES \
  { 704, }
#define AI_SURFACE_NN_DATA_ACTIVATIONS_SIZE     (704)
#define AI_SURFACE_NN_DATA_ACTIVATIONS_COUNT    (1)
#define AI_SURFACE_NN_DATA_ACTIVATION_1_SIZE    (704)



#define AI_SURFACE_NN_DATA_WEIGHTS_SIZES \
  { 6160, }
#define AI_SURFACE_NN_DATA_WEIGHTS_SIZE         (6160)
#define AI_SURFACE_NN_DATA_WEIGHTS_COUNT        (1)
#define AI_SURFACE_NN_DATA_WEIGHT_1_SIZE        (6160)



#define AI_SURFACE_NN_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_surface_nn_activations_table[1])

extern ai_handle g_surface_nn_activations_table[1 + 2];



#define AI_SURFACE_NN_DATA_WEIGHTS_TABLE_GET() \
  (&g_surface_nn_weights_table[1])

extern ai_handle g_surface_nn_weights_table[1 + 2];


#endif    /* SURFACE_NN_DATA_PARAMS_H */
