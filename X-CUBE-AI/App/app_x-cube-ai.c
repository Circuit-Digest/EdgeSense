/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  EdgeSense Embedded Team / STMicroelectronics X-CUBE-AI
  * @brief   EdgeSense STM32Cube.AI Middleware Bridge & Lifecycle Hooks
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "main.h"
#include "app_x-cube-ai.h"
#include "pipeline_manager.h"
#include "model_gesture.h"

void MX_X_CUBE_AI_Init(void)
{
    // Initialize the unified multi-capability pipeline
    Pipeline_Init();
}

void MX_X_CUBE_AI_Process(void)
{
    Pipeline_Process();
}

void EdgeSense_AI_FeedFrame(const VL53LMZ_ResultsData *p_res)
{
    Pipeline_FeedFrame(p_res, NULL, 0);
}

const gesture_nn_result_t* EdgeSense_AI_GetLatestResult(void)
{
    static gesture_nn_result_t bridge_res = {0};
    const Pipeline_GestureResult_t *g = Pipeline_GetLatestGesture();
    if (g != NULL) {
        bridge_res.detected_class = (gesture_nn_class_t)g->detected_class;
        bridge_res.confidence = g->confidence;
        bridge_res.latency_us = g->latency_us;
        bridge_res.is_new_event = g->is_new_event;
        for (int i = 0; i < 5; i++) {
            bridge_res.probabilities[i] = g->probabilities[i];
        }
    }
    return &bridge_res;
}

const char* EdgeSense_AI_GetClassName(gesture_nn_class_t cls)
{
    return Model_Gesture_GetClassName((uint8_t)cls);
}
