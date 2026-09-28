/*
 * pipeline_manager.h
 * Header for Unified Multi-Capability AI Pipeline Manager
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
  * @file    pipeline_manager.h
  * @brief   Unified Multi-Capability AI Pipeline Manager Header
  ******************************************************************************
  */

#ifndef __PIPELINE_MANAGER_H
#define __PIPELINE_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "vl53lmz_api.h"
#include "ai_platform.h"

// 20 KB shared activation buffer for all embedded neural models
#define AI_SHARED_ACTIVATION_BUFFER_SIZE 20480

typedef enum {
    SENSE_MODE_GESTURE = 0,
    SENSE_MODE_POSTURE = 1,
    SENSE_MODE_SURFACE = 2,
    SENSE_MODE_SMOKE   = 3,
    SENSE_MODE_COUNT
} SenseMode_t;

typedef enum {
    STREAM_MODE_RESULTS = 0,
    STREAM_MODE_RAW     = 1,
    STREAM_MODE_DEBUG   = 2,
    STREAM_MODE_OFF     = 3
} StreamMode_t;

// Gesture Recognition Result Structure
typedef struct {
    uint8_t detected_class;
    float confidence;
    float probabilities[5];
    uint32_t latency_us;
    bool is_new_event;
} Pipeline_GestureResult_t;

// Hand Posture Result Structure
typedef struct {
    uint8_t detected_posture;
    float confidence;
    uint32_t latency_us;
    bool is_valid_hand;
    bool is_new_event;
} Pipeline_PostureResult_t;

// Surface Classification Result Structure
typedef struct {
    uint8_t surface_type;
    float confidence;
    float pulse_fwhm_mm;
    float peak_signal;
    float reflectance_index;
    uint32_t latency_us;
    bool is_new_event;
} Pipeline_SurfaceResult_t;

// Smoke & Obscuration Detection Result Structure
typedef struct {
    uint8_t alert_level;
    float obscuration_pct;
    float attenuation_ratio;
    uint32_t latency_us;
    bool alarm_active;
} Pipeline_SmokeResult_t;

// Lifecycle and control API
void Pipeline_Init(void);
void Pipeline_Process(void);

// Set / Get active pipeline mode
bool Pipeline_SetMode(SenseMode_t mode);
SenseMode_t Pipeline_GetMode(void);
const char* Pipeline_GetModeName(SenseMode_t mode);

// Set / Get streaming mode
void Pipeline_SetStreamMode(StreamMode_t mode);
StreamMode_t Pipeline_GetStreamMode(void);
const char* Pipeline_GetStreamModeName(StreamMode_t mode);

// Feed frame from sensor acquisition loop (15.0 FPS)
void Pipeline_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes);

// Query latest results for active mode
const Pipeline_GestureResult_t* Pipeline_GetLatestGesture(void);
const Pipeline_PostureResult_t* Pipeline_GetLatestPosture(void);
const Pipeline_SurfaceResult_t* Pipeline_GetLatestSurface(void);
const Pipeline_SmokeResult_t* Pipeline_GetLatestSmoke(void);

// Microsecond execution time for latest inference
uint32_t Pipeline_GetLatestLatencyUs(void);

// Global shared activation RAM pointer for sub-models
uint8_t* Pipeline_GetSharedActivationPool(void);

#ifdef __cplusplus
}
#endif

#endif /* __PIPELINE_MANAGER_H */
