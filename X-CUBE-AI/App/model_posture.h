/*
 * model_posture.h
 * Static Hand Posture Classification Module Header
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
  * @file    model_posture.h
  * @brief   Static Hand Posture Classification Module Header (STSW-IMG050 aligned)
  ******************************************************************************
  */

#ifndef __MODEL_POSTURE_H
#define __MODEL_POSTURE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "pipeline_manager.h"
#include "vl53lmz_api.h"

#define POSTURE_CLASSES_COUNT 6
#define POSTURE_MIN_DISTANCE  100.0f
#define POSTURE_MAX_DISTANCE  400.0f
#define POSTURE_BG_REMOVAL    120.0f

typedef enum {
    POSTURE_CLASS_NONE = 0,
    POSTURE_CLASS_FLAT_HAND = 1,
    POSTURE_CLASS_LIKE = 2,
    POSTURE_CLASS_DISLIKE = 3,
    POSTURE_CLASS_BREAK_TIME = 4,
    POSTURE_CLASS_FIST = 5
} PostureClass_t;

bool Model_Posture_Init(void);
void Model_Posture_FeedFrame(const VL53LMZ_ResultsData *p_res);
void Model_Posture_Reset(void);
const Pipeline_PostureResult_t* Model_Posture_GetResult(void);
const char* Model_Posture_GetClassName(uint8_t cls);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_POSTURE_H */
