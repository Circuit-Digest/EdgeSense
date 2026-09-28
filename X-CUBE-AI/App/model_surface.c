/*
 * model_surface.c
 * Distance-Invariant Surface and Material Classification Engine (1D-CNN)
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
  * @file    model_surface.c
  * @brief   Distance-Invariant Surface / Material 1D-CNN Classification (CNH)
  *          Features: 4x4 CNH histogram binning, 3-tap binomial smoothing,
  *                    peak-relative canonical alignment, 16-zone spatial voting,
  *                    and 3-frame state transition hysteresis.
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "main.h"
#include "model_surface.h"
#include "vl53lmz_plugin_cnh.h"
#include "surface_nn.h"
#include "surface_nn_data.h"

static const char *s_surface_names[SURFACE_CLASSES_COUNT] = {
    "HARD_FLOOR (Tile / Wood / Ceramic)",
    "CARPET (Fabric / Rug / Dispersion)",
    "SPECULAR (Mirror / Metallic)",
    "VOID (Drop-Off / Cliff / Air)"
};

static ai_handle s_surface_network = AI_HANDLE_NULL;
static ai_buffer *s_ai_input = NULL;
static ai_buffer *s_ai_output = NULL;
static bool s_ai_ready = false;

// 704-byte dedicated activation buffer aligned to 32 bytes
AI_ALIGNED(32)
static uint8_t s_surface_activations[AI_SURFACE_NN_DATA_ACTIVATIONS_SIZE];

static Pipeline_SurfaceResult_t s_latest_result = {0};
static uint8_t s_confirmed_class = SURFACE_TYPE_VOID;
static uint8_t s_candidate_class = SURFACE_TYPE_VOID;
static uint8_t s_candidate_count = 0;
static float s_smoothed_probs[SURFACE_CLASSES_COUNT] = {0.25f, 0.25f, 0.25f, 0.25f};

static void smooth_cnh(const float *raw, float *smoothed, int n)
{
    for (int i = 0; i < n; i++) {
        float p_prev = (i > 0) ? raw[i - 1] : raw[i];
        float p_curr = raw[i];
        float p_next = (i < n - 1) ? raw[i + 1] : raw[i];
        smoothed[i] = 0.25f * p_prev + 0.50f * p_curr + 0.25f * p_next;
    }
}

static void softmax4(const float *logits, float *probs)
{
    float max_l = logits[0];
    for (int i = 1; i < 4; i++) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        probs[i] = expf(logits[i] - max_l);
        sum += probs[i];
    }
    float inv_sum = (sum > 1e-6f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < 4; i++) {
        probs[i] *= inv_sum;
    }
}

bool Model_Surface_Init(void)
{
    // Enable Cortex-M33 DWT cycle counter for microsecond benchmark
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    ai_handle act_addr[] = { s_surface_activations };
    ai_error err = ai_surface_nn_create_and_init(&s_surface_network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("[SURFACE-ERR] ai_surface_nn_create_and_init failed: %d\r\n", err.type);
        s_ai_ready = false;
        return false;
    }

    ai_u16 n_in = 0, n_out = 0;
    s_ai_input = ai_surface_nn_inputs_get(s_surface_network, &n_in);
    s_ai_output = ai_surface_nn_outputs_get(s_surface_network, &n_out);
    s_ai_ready = true;
    Model_Surface_Reset();
    printf("[AI OK] Model 'surface_nn' (1D-CNN) loaded successfully (%d bytes RAM).\r\n",
           AI_SURFACE_NN_DATA_ACTIVATIONS_SIZE);
    return true;
}

void Model_Surface_Reset(void)
{
    memset(&s_latest_result, 0, sizeof(s_latest_result));
    s_confirmed_class = SURFACE_TYPE_VOID;
    s_candidate_class = SURFACE_TYPE_VOID;
    s_candidate_count = 0;
    s_smoothed_probs[0] = 0.0f;
    s_smoothed_probs[1] = 0.0f;
    s_smoothed_probs[2] = 0.0f;
    s_smoothed_probs[3] = 1.0f; // Default VOID
}

void Model_Surface_FeedFrame(const VL53LMZ_ResultsData *p_res, const uint32_t *p_cnh, uint32_t cnh_bytes)
{
    if (p_res == NULL) return;

    uint32_t t_start = DWT->CYCCNT;

    int total_zones = (int)Get_Sensor_Resolution();
    int num_bins = (int)Get_Sensor_NumBins();

    // Fallback if CNH buffer is empty or model is not loaded
    if (!s_ai_ready || p_cnh == NULL || cnh_bytes < 64 || num_bins < 16) {
        s_latest_result.surface_type = SURFACE_TYPE_VOID;
        s_latest_result.confidence = 0.99f;
        s_latest_result.pulse_fwhm_mm = 0.0f;
        s_latest_result.latency_us = 0;
        s_latest_result.is_new_event = false;
        return;
    }

    float ensemble_probs[SURFACE_CLASSES_COUNT] = {0};
    int valid_zones_count = 0;
    float sum_fwhm_mm = 0.0f;
    float sum_signal = 0.0f;
    float sum_refl = 0.0f;

    // Buffer for canonical 16-bin pulse
    float canonical[16];
    float raw_pulse[48];
    float smoothed[48];
    float zone_logits[4];
    float zone_probs[4];

    VL53LMZ_Motion_Configuration *p_cnh_cfg = (VL53LMZ_Motion_Configuration *)Get_CNH_Configuration();

    for (int z = 0; z < total_zones && z < 16; z++) {
        int16_t d = (int16_t)p_res->distance_mm[z];

        if (d <= 0) {
            // Zone sees void / drop-off (inject soft void distribution aligned with Python Studio)
            ensemble_probs[SURFACE_TYPE_HARD_FLOOR] += 0.01f;
            ensemble_probs[SURFACE_TYPE_CARPET]     += 0.01f;
            ensemble_probs[SURFACE_TYPE_SPECULAR]   += 0.01f;
            ensemble_probs[SURFACE_TYPE_VOID]       += 0.97f;
            valid_zones_count++;
            continue;
        }

        int32_t *p_hist = NULL;
        int8_t *p_hist_scaler = NULL;
        int32_t *p_ambient = NULL;
        int8_t *p_ambient_scaler = NULL;

        vl53lmz_cnh_get_block_addresses(p_cnh_cfg, z, (uint32_t *)p_cnh,
                                        &p_hist, &p_hist_scaler, &p_ambient, &p_ambient_scaler);

        if (!p_hist || !p_hist_scaler) {
            ensemble_probs[SURFACE_TYPE_HARD_FLOOR] += 0.01f;
            ensemble_probs[SURFACE_TYPE_CARPET]     += 0.01f;
            ensemble_probs[SURFACE_TYPE_SPECULAR]   += 0.01f;
            ensemble_probs[SURFACE_TYPE_VOID]       += 0.97f;
            valid_zones_count++;
            continue;
        }

        // Unpack raw pulse
        int bins_to_process = (num_bins > 48) ? 48 : num_bins;
        for (int b = 0; b < bins_to_process; b++) {
            raw_pulse[b] = ((float)p_hist[b]) / (float)(1 << p_hist_scaler[b]);
        }
        for (int b = bins_to_process; b < 48; b++) {
            raw_pulse[b] = 0.0f;
        }

        // ST 3-tap binomial smoothing
        smooth_cnh(raw_pulse, smoothed, 48);

        // Find peak and total energy
        float peak_val = 0.0f;
        int peak_bin = 0;
        float total_energy = 0.0f;
        for (int b = 0; b < 48; b++) {
            total_energy += smoothed[b];
            if (smoothed[b] > peak_val) {
                peak_val = smoothed[b];
                peak_bin = b;
            }
        }

        // Check if void / drop-off
        if (peak_val < 8.0f || total_energy < 40.0f) {
            ensemble_probs[SURFACE_TYPE_HARD_FLOOR] += 0.01f;
            ensemble_probs[SURFACE_TYPE_CARPET]     += 0.01f;
            ensemble_probs[SURFACE_TYPE_SPECULAR]   += 0.01f;
            ensemble_probs[SURFACE_TYPE_VOID]       += 0.97f;
            valid_zones_count++;
            continue;
        }

        // Sub-bin interpolated FWHM in mm (37.46 mm / bin)
        float half_val = peak_val * 0.50f;
        float rise_pos = (float)peak_bin;
        for (int pos = peak_bin; pos >= 1; pos--) {
            if (smoothed[pos] >= half_val && smoothed[pos - 1] <= half_val) {
                float y0 = smoothed[pos - 1];
                float y1 = smoothed[pos];
                float frac = (half_val - y0) / ((y1 - y0) > 1e-3f ? (y1 - y0) : 1e-3f);
                rise_pos = (float)(pos - 1) + frac;
                break;
            }
        }
        float fall_pos = (float)(peak_bin + 1);
        for (int pos = peak_bin; pos < 47; pos++) {
            if (smoothed[pos] >= half_val && smoothed[pos + 1] <= half_val) {
                float y0 = smoothed[pos];
                float y1 = smoothed[pos + 1];
                float frac = (half_val - y0) / ((y1 - y0) < -1e-3f ? (y1 - y0) : -1e-3f);
                fall_pos = (float)pos + frac;
                break;
            }
        }
        float fwhm_bins = fall_pos - rise_pos;
        if (fwhm_bins < 0.5f) fwhm_bins = 0.5f;
        float fwhm_mm = fwhm_bins * 37.46f;
        sum_fwhm_mm += fwhm_mm;

        sum_signal += (float)p_res->signal_per_spad[z] / 2048.0f;
        sum_refl += (float)p_res->reflectance[z];

        // Extract 16-bin canonical window centered on peak: [peak_bin - 4 ... peak_bin + 11]
        int start_idx = peak_bin - 4;
        for (int i = 0; i < 16; i++) {
            int src = start_idx + i;
            if (src >= 0 && src < 48) {
                canonical[i] = smoothed[src];
            } else {
                canonical[i] = 0.0f;
            }
        }

        // Peak normalize so canonical[4] == 1.0f
        if (peak_val > 1e-3f) {
            for (int i = 0; i < 16; i++) {
                canonical[i] /= peak_val;
            }
        }

        // Setup input tensor (fallback to direct pointer if needed)
        if (s_ai_input[0].data != NULL) {
            memcpy(s_ai_input[0].data, canonical, 16 * sizeof(float));
        } else {
            s_ai_input[0].data = AI_HANDLE_PTR(canonical);
        }
        if (s_ai_output[0].data == NULL) {
            s_ai_output[0].data = AI_HANDLE_PTR(zone_logits);
        }

        // Run Cube.AI 1D-CNN Inference
        ai_i32 batch = ai_surface_nn_run(s_surface_network, s_ai_input, s_ai_output);
        if (batch > 0) {
            memcpy(zone_logits, (float *)s_ai_output[0].data, 4 * sizeof(float));
            softmax4(zone_logits, zone_probs);
            for (int c = 0; c < 4; c++) {
                ensemble_probs[c] += zone_probs[c];
            }
        } else {
            ensemble_probs[SURFACE_TYPE_HARD_FLOOR] += 0.01f;
            ensemble_probs[SURFACE_TYPE_CARPET]     += 0.01f;
            ensemble_probs[SURFACE_TYPE_SPECULAR]   += 0.01f;
            ensemble_probs[SURFACE_TYPE_VOID]       += 0.97f;
        }

        valid_zones_count++;
    }

    if (valid_zones_count == 0) return;

    // 1. Average ensemble probabilities across all zones
    float inv_zones = 1.0f / (float)valid_zones_count;
    for (int c = 0; c < 4; c++) {
        ensemble_probs[c] *= inv_zones;
    }

    // 2. Temporal Exponential Moving Average (EMA: alpha=0.30)
    const float alpha = 0.30f;
    for (int c = 0; c < 4; c++) {
        s_smoothed_probs[c] = (1.0f - alpha) * s_smoothed_probs[c] + alpha * ensemble_probs[c];
    }

    // 3. Find winning candidate class
    int best_class = 0;
    float best_prob = s_smoothed_probs[0];
    for (int c = 1; c < 4; c++) {
        if (s_smoothed_probs[c] > best_prob) {
            best_prob = s_smoothed_probs[c];
            best_class = c;
        }
    }

    // 4. 3-Frame State Transition Debounce Hysteresis
    bool is_new = false;
    if (best_class != s_confirmed_class) {
        if (best_class == s_candidate_class && best_prob >= 0.55f) {
            s_candidate_count++;
            if (s_candidate_count >= 3) {
                s_confirmed_class = best_class;
                s_candidate_count = 0;
                is_new = true;
            }
        } else {
            s_candidate_class = best_class;
            s_candidate_count = 1;
        }
    } else {
        s_candidate_count = 0;
    }

    uint32_t t_end = DWT->CYCCNT;
    uint32_t cycles = t_end - t_start;
    uint32_t latency_us = (uint32_t)((uint64_t)cycles * 1000000ULL / SystemCoreClock);

    s_latest_result.surface_type = s_confirmed_class;
    s_latest_result.confidence = best_prob;
    s_latest_result.pulse_fwhm_mm = sum_fwhm_mm * inv_zones;
    s_latest_result.peak_signal = sum_signal * inv_zones;
    s_latest_result.reflectance_index = sum_refl * inv_zones;
    s_latest_result.latency_us = latency_us;
    s_latest_result.is_new_event = is_new;

    if (is_new) {
        printf("[AI SURFACE] >>> DETECTED: %s (Confidence: %.1f%% | FWHM: %.1f mm | Latency: %lu us) <<<\r\n",
               s_surface_names[s_confirmed_class], (double)(best_prob * 100.0f),
               (double)s_latest_result.pulse_fwhm_mm, (unsigned long)latency_us);
    }
}

const Pipeline_SurfaceResult_t* Model_Surface_GetResult(void)
{
    return &s_latest_result;
}

const char* Model_Surface_GetTypeName(uint8_t type)
{
    if (type < SURFACE_CLASSES_COUNT) return s_surface_names[type];
    return "Unknown";
}
