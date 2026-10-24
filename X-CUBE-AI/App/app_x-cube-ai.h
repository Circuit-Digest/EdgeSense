
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_x-cube-ai.h
  * @author  EdgeSense Embedded Team / STMicroelectronics X-CUBE-AI
  * @brief   EdgeSense STM32Cube.AI Real-Time Gesture Recognition Interface
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

#ifndef __APP_AI_H
#define __APP_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "ai_platform.h"
#include "vl53lmz_api.h"

#define GESTURE_NN_CLASSES_COUNT 5
#define GESTURE_NN_WINDOW_SIZE   15
#define GESTURE_NN_FEATURE_DIM   241

typedef enum {
    GESTURE_NN_IDLE = 0,
    GESTURE_NN_SWIPE_LEFT = 1,
    GESTURE_NN_SWIPE_RIGHT = 2,
    GESTURE_NN_SWIPE_UP = 3,
    GESTURE_NN_SWIPE_DOWN = 4
} gesture_nn_class_t;

typedef struct {
    gesture_nn_class_t detected_class;
    float confidence;
    float probabilities[GESTURE_NN_CLASSES_COUNT];
    uint32_t latency_us;
    uint32_t cycle_count;
    bool is_new_event;
} gesture_nn_result_t;

// Lifecycle entry points
void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);

// Feed a new 64-zone frame from VL53L8CH (called in sensor acquisition loop)
void EdgeSense_AI_FeedFrame(const VL53LMZ_ResultsData *p_res);

// Query latest inference result
const gesture_nn_result_t* EdgeSense_AI_GetLatestResult(void);

// Convert class enum to human-readable string
const char* EdgeSense_AI_GetClassName(gesture_nn_class_t cls);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */

