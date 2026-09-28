/*
 * pipeline_manager.c
 * Unified Multi-Capability AI Pipeline Manager Implementation
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
  * @file    pipeline_manager.c
  * @brief   Unified Multi-Capability AI Pipeline Manager
  *          Coordinates execution of Gesture, Posture, Surface, and Smoke models
  *          over shared memory scratchpad with zero fragmentation.
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "main.h"
#include "pipeline_manager.h"
#include "model_gesture.h"
#include "model_posture.h"
#include "model_surface.h"
#include "model_smoke.h"

// 20 KB shared activation buffer aligned to 32 bytes
AI_ALIGNED(32)
static uint8_t s_shared_activations[AI_SHARED_ACTIVATION_BUFFER_SIZE];

static SenseMode_t s_active_mode = SENSE_MODE_GESTURE;
static StreamMode_t s_stream_mode = STREAM_MODE_RESULTS;

static const char *s_mode_names[SENSE_MODE_COUNT] = {
    "GESTURE", "POSTURE", "SURFACE", "SMOKE"
};

static const char *s_stream_names[4] = {
    "RESULTS", "RAW", "DEBUG", "OFF"
};

void Pipeline_Init(void)
{
    printf("\r\n==============================================\r\n");
    printf("--- Initializing EdgeSense AI Pipeline Manager ---\r\n");
    printf("Shared Activation RAM: %d bytes @ 0x%08lX\r\n", 
           AI_SHARED_ACTIVATION_BUFFER_SIZE, (uint32_t)s_shared_activations);

    Model_Gesture_Init();
    Model_Posture_Init();
    Model_Surface_Init();
    Model_Smoke_Init();

    s_active_mode = SENSE_MODE_GESTURE;
    s_stream_mode = STREAM_MODE_RESULTS;
    printf("[PIPELINE] Default Mode: GESTURE | Stream: RESULTS\r\n");
    printf("==============================================\r\n");
}

void Pipeline_Process(void)
{
    // Background tasks if any (non-blocking)
}

bool Pipeline_SetMode(SenseMode_t mode)
{
    if (mode >= SENSE_MODE_COUNT) return false;
    if (mode == s_active_mode) return true;

    s_active_mode = mode;
    switch (mode) {
        case SENSE_MODE_GESTURE:
            Model_Gesture_Reset();
            break;
        case SENSE_MODE_POSTURE:
            Model_Posture_Reset();
            break;
        case SENSE_MODE_SURFACE:
            Model_Surface_Reset();
            break;
        case SENSE_MODE_SMOKE:
            Model_Smoke_Reset();
            break;
        default:
            break;
    }
    printf("[PIPELINE] Active mode switched to: %s\r\n", s_mode_names[mode]);
    return true;
}

SenseMode_t Pipeline_GetMode(void)
{
    return s_active_mode;
}

const char* Pipeline_GetModeName(SenseMode_t mode)
{
    if (mode < SENSE_MODE_COUNT) return s_mode_names[mode];
    return "UNKNOWN";
}

void Pipeline_SetStreamMode(StreamMode_t mode)
{
    if (mode <= STREAM_MODE_OFF) {
        s_stream_mode = mode;
        printf("[PIPELINE] Stream mode set to: %s\r\n", s_stream_names[mode]);
    }
}

StreamMode_t Pipeline_GetStreamMode(void)
{
    return s_stream_mode;
}

const char* Pipeline_GetStreamModeName(StreamMode_t mode)
{
    if (mode <= STREAM_MODE_OFF) return s_stream_names[mode];
    return "UNKNOWN";
}

uint8_t* Pipeline_GetSharedActivationPool(void)
{
    return s_shared_activations;
}

void Pipeline_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes)
{
    if (p_res == NULL) return;

    switch (s_active_mode) {
        case SENSE_MODE_GESTURE:
            Model_Gesture_FeedFrame(p_res);
            break;
        case SENSE_MODE_POSTURE:
            Model_Posture_FeedFrame(p_res);
            break;
        case SENSE_MODE_SURFACE:
            Model_Surface_FeedFrame(p_res, p_cnh, cnh_bytes);
            break;
        case SENSE_MODE_SMOKE:
            Model_Smoke_FeedFrame(p_res, p_cnh, cnh_bytes);
            break;
        default:
            break;
    }
}

const Pipeline_GestureResult_t* Pipeline_GetLatestGesture(void)
{
    return Model_Gesture_GetResult();
}

const Pipeline_PostureResult_t* Pipeline_GetLatestPosture(void)
{
    return Model_Posture_GetResult();
}

const Pipeline_SurfaceResult_t* Pipeline_GetLatestSurface(void)
{
    return Model_Surface_GetResult();
}

const Pipeline_SmokeResult_t* Pipeline_GetLatestSmoke(void)
{
    return Model_Smoke_GetResult();
}

uint32_t Pipeline_GetLatestLatencyUs(void)
{
    switch (s_active_mode) {
        case SENSE_MODE_GESTURE:
            return Model_Gesture_GetResult()->latency_us;
        case SENSE_MODE_POSTURE:
            return Model_Posture_GetResult()->latency_us;
        case SENSE_MODE_SURFACE:
            return Model_Surface_GetResult()->latency_us;
        case SENSE_MODE_SMOKE:
            return Model_Smoke_GetResult()->latency_us;
        default:
            return 0;
    }
}
