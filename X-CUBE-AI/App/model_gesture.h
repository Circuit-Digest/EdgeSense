/*
 * model_gesture.h
 * Header for 3D Dynamic Gesture Recognition Module
 *
 * EdgeSense: STM32 Edge AI Texture and Gesture Classifier
 *
 * Copyright (c) 2026 Dharagesh and Circuit Digest
 * https://github.com/Circuit-Digest/EdgeSense
 * Licensed under GNU General Public License v3.0
 */

/*
  ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
  ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
*/
/**
  ******************************************************************************
  * @file    model_gesture.h
  * @brief   3D Dynamic Gesture Recognition Model Module Header
  ******************************************************************************
  */

#ifndef __MODEL_GESTURE_H
#define __MODEL_GESTURE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "pipeline_manager.h"
#include "vl53lmz_api.h"

#define GESTURE_CLASSES_COUNT 5
#define GESTURE_WINDOW_SIZE   15
#define GESTURE_GRID_SIZE     8
#define GESTURE_INPUT_DIM     (GESTURE_WINDOW_SIZE * GESTURE_GRID_SIZE * GESTURE_GRID_SIZE)

typedef enum {
    GESTURE_CLASS_IDLE = 0,
    GESTURE_CLASS_SWIPE_LEFT = 1,
    GESTURE_CLASS_SWIPE_RIGHT = 2,
    GESTURE_CLASS_SWIPE_UP = 3,
    GESTURE_CLASS_SWIPE_DOWN = 4
} GestureClass_t;

typedef enum {
    GESTURE_FSM_IDLE = 0,
    GESTURE_FSM_TRACKING = 1,
    GESTURE_FSM_EVALUATING = 2,
    GESTURE_FSM_EXIT_LOCKOUT = 3
} GestureFsmState_t;

bool Model_Gesture_Init(void);
void Model_Gesture_FeedFrame(const VL53LMZ_ResultsData *p_res);
void Model_Gesture_Reset(void);
const Pipeline_GestureResult_t* Model_Gesture_GetResult(void);
const char* Model_Gesture_GetClassName(uint8_t cls);
GestureFsmState_t Model_Gesture_GetFsmState(void);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_GESTURE_H */
