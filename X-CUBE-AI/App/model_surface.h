/*
 * model_surface.h
 * Surface and Material Classification Module Header (CNH Histogram)
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
  * @file    model_surface.h
  * @brief   Surface / Material Classification Module Header (CNH Histogram)
  ******************************************************************************
  */

#ifndef __MODEL_SURFACE_H
#define __MODEL_SURFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "pipeline_manager.h"
#include "vl53lmz_api.h"

#define SURFACE_CLASSES_COUNT 4

typedef enum {
    SURFACE_TYPE_HARD_FLOOR = 0,
    SURFACE_TYPE_CARPET     = 1,
    SURFACE_TYPE_SPECULAR   = 2,
    SURFACE_TYPE_VOID       = 3
} SurfaceType_t;

bool Model_Surface_Init(void);
void Model_Surface_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes);
void Model_Surface_Reset(void);
const Pipeline_SurfaceResult_t* Model_Surface_GetResult(void);
const char* Model_Surface_GetTypeName(uint8_t type);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_SURFACE_H */
