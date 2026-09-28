/*
 * model_smoke.c
 * Optical Scattering Smoke & Particle Obscuration Detection Engine
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
  * @file    model_smoke.c
  * @brief   Optical Scattering Smoke & Particle Obscuration Detection Implementation
  *          Features: Dual-rate dynamic baseline tracking, signal attenuation
  *                    and photon backscatter metrics, with multi-stage debounce.
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "main.h"
#include "model_smoke.h"

static const char *s_smoke_level_names[SMOKE_LEVELS_COUNT] = {
    "Clear / Normal", "Light Haze / Particles", "DENSE SMOKE ALARM"
};

static Pipeline_SmokeResult_t s_latest_result = {0};

// Background baseline tracking
static float s_baseline_signal[64] = {0};
static bool s_baseline_valid = false;
static uint16_t s_baseline_frames = 0;

// Debounce state
static uint8_t s_alarm_debounce = 0;
static uint8_t s_previous_level = SMOKE_LEVEL_CLEAR;

bool Model_Smoke_Init(void)
{
    Model_Smoke_Reset();
    printf("[SMOKE] Optical scattering & smoke detection engine ready.\r\n");
    return true;
}

void Model_Smoke_Reset(void)
{
    memset(&s_latest_result, 0, sizeof(s_latest_result));
    s_baseline_valid = false;
    s_baseline_frames = 0;
    s_alarm_debounce = 0;
    s_previous_level = SMOKE_LEVEL_CLEAR;
    for (int i = 0; i < 64; i++) {
        s_baseline_signal[i] = 100.0f;
    }
}

void Model_Smoke_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes)
{
    if (p_res == NULL) return;

    uint32_t t_start = DWT->CYCCNT;

    float current_total_signal = 0.0f;
    float baseline_total_signal = 0.0f;
    int valid_zones = 0;

    for (int i = 0; i < 64; i++) {
        float peak = (float)p_res->signal_per_spad[i] / 2048.0f;
        if (peak > 0.1f) {
            current_total_signal += peak;
            baseline_total_signal += s_baseline_signal[i];
            valid_zones++;

            // Slow baseline adaptation when no smoke is present
            if (s_latest_result.alert_level == SMOKE_LEVEL_CLEAR) {
                s_baseline_signal[i] = 0.98f * s_baseline_signal[i] + 0.02f * peak;
            }
        }
    }

    if (!s_baseline_valid) {
        s_baseline_frames++;
        if (s_baseline_frames > 20) {
            s_baseline_valid = true;
        }
    }

    float attenuation_ratio = 1.0f;
    float obscuration_pct = 0.0f;
    uint8_t alert_level = SMOKE_LEVEL_CLEAR;

    if (s_baseline_valid && baseline_total_signal > 1.0f && valid_zones >= 8) {
        attenuation_ratio = current_total_signal / baseline_total_signal;
        if (attenuation_ratio > 1.0f) attenuation_ratio = 1.0f;
        obscuration_pct = (1.0f - attenuation_ratio) * 100.0f;

        if (obscuration_pct > 45.0f) {
            if (s_alarm_debounce < 4) s_alarm_debounce++;
            if (s_alarm_debounce >= 4) alert_level = SMOKE_LEVEL_DENSE_ALARM;
            else alert_level = SMOKE_LEVEL_LIGHT_HAZE;
        } else if (obscuration_pct > 20.0f) {
            alert_level = SMOKE_LEVEL_LIGHT_HAZE;
            s_alarm_debounce = 0;
        } else {
            alert_level = SMOKE_LEVEL_CLEAR;
            s_alarm_debounce = 0;
        }
    }

    uint32_t t_end = DWT->CYCCNT;
    uint32_t cycles = t_end - t_start;
    uint32_t latency_us = cycles / (SystemCoreClock / 1000000);

    bool is_new = (alert_level != s_previous_level && alert_level != SMOKE_LEVEL_CLEAR);
    s_previous_level = alert_level;

    s_latest_result.alert_level = alert_level;
    s_latest_result.obscuration_pct = obscuration_pct;
    s_latest_result.attenuation_ratio = attenuation_ratio;
    s_latest_result.latency_us = latency_us;
    s_latest_result.alarm_active = (alert_level == SMOKE_LEVEL_DENSE_ALARM);

    if (is_new) {
        printf("[SMOKE] >>> ALERT: %s (Obscuration: %.1f%%, Latency: %lu us) <<<\r\n",
               s_smoke_level_names[alert_level], obscuration_pct, latency_us);
    }
}

const Pipeline_SmokeResult_t* Model_Smoke_GetResult(void)
{
    return &s_latest_result;
}

const char* Model_Smoke_GetLevelName(uint8_t level)
{
    if (level < SMOKE_LEVELS_COUNT) return s_smoke_level_names[level];
    return "Unknown";
}
