/*
 * model_smoke.h
 * Optical Scattering Smoke & Particle Obscuration Detection Module Header
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
  * @file    model_smoke.h
  * @brief   Optical Scattering Smoke & Particle Obscuration Detection Header
  ******************************************************************************
  */

#ifndef __MODEL_SMOKE_H
#define __MODEL_SMOKE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "pipeline_manager.h"
#include "vl53lmz_api.h"

#define SMOKE_LEVELS_COUNT 3

typedef enum {
    SMOKE_LEVEL_CLEAR = 0,
    SMOKE_LEVEL_LIGHT_HAZE = 1,
    SMOKE_LEVEL_DENSE_ALARM = 2
} SmokeLevel_t;

bool Model_Smoke_Init(void);
void Model_Smoke_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes);
void Model_Smoke_Reset(void);
const Pipeline_SmokeResult_t* Model_Smoke_GetResult(void);
const char* Model_Smoke_GetLevelName(uint8_t level);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_SMOKE_H */
