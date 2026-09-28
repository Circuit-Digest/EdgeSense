/*
 * model_posture.c
 * Static Hand Posture Classification Engine (STSW-IMG050 2D-CNN Aligned)
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
  * @file    model_posture.c
  * @brief   Static Hand Posture Classification Engine
  *          Features: Planar NCHW (1, 2, 8, 8) input (Distance + Signal channels),
  *                    RobustScaler normalization & ST background segmentation,
  *                    3-frame temporal debounce latching,
  *                    Classes: NONE, FLAT_HAND, LIKE, DISLIKE, BREAK_TIME, FIST
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "main.h"
#include "model_posture.h"
#include "posture_nn.h"
#include "posture_nn_data.h"

static const char *s_posture_names[POSTURE_CLASSES_COUNT] = {
    "NONE", "FLAT_HAND", "LIKE", "DISLIKE", "BREAK_TIME", "FIST"
};

// Normalization constants (ST RobustScaler)
#define NORMALIZATION_RANGING_CENTER 295.0f
#define NORMALIZATION_RANGING_IQR    196.0f
#define NORMALIZATION_SIGNAL_CENTER  281.0f
#define NORMALIZATION_SIGNAL_IQR     452.0f

#define DEFAULT_RANGING_VALUE        4000.0f
#define DEFAULT_SIGNAL_VALUE         0.0f

#define POSTURE_CONFIDENCE_THRESH    0.70f
#define POSTURE_FILTER_N             3

static ai_handle s_posture_network = AI_HANDLE_NULL;
static ai_buffer *s_ai_input = NULL;
static ai_buffer *s_ai_output = NULL;
static bool s_ai_ready = false;

// 1096-byte dedicated activation buffer aligned to 32 bytes
AI_ALIGNED(32)
static uint8_t s_posture_activations[AI_POSTURE_NN_DATA_ACTIVATIONS_SIZE];

static Pipeline_PostureResult_t s_latest_result = {0};

// Temporal filter state (requires 3 consecutive frames to confirm posture)
static uint8_t s_previous_label = POSTURE_CLASS_NONE;
static uint8_t s_label_count = 0;
static uint8_t s_none_count = 0;
static uint8_t s_latched_posture = POSTURE_CLASS_NONE;

static void softmax6(const float *logits, float *probs)
{
    float max_l = logits[0];
    for (int i = 1; i < POSTURE_CLASSES_COUNT; i++) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < POSTURE_CLASSES_COUNT; i++) {
        probs[i] = expf(logits[i] - max_l);
        sum += probs[i];
    }
    float inv_sum = (sum > 1e-6f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < POSTURE_CLASSES_COUNT; i++) {
        probs[i] *= inv_sum;
    }
}

bool Model_Posture_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    ai_handle act_addr[] = { s_posture_activations };
    ai_error err = ai_posture_nn_create_and_init(&s_posture_network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("[POSTURE-ERR] ai_posture_nn_create_and_init failed: %d\r\n", err.type);
        s_ai_ready = false;
        return false;
    }

    ai_u16 n_in = 0, n_out = 0;
    s_ai_input = ai_posture_nn_inputs_get(s_posture_network, &n_in);
    s_ai_output = ai_posture_nn_outputs_get(s_posture_network, &n_out);
    s_ai_ready = true;
    Model_Posture_Reset();
    printf("[AI OK] Model 'posture_nn' (2D-CNN) loaded successfully (%d bytes RAM).\r\n",
           AI_POSTURE_NN_DATA_ACTIVATIONS_SIZE);
    return true;
}

void Model_Posture_Reset(void)
{
    memset(&s_latest_result, 0, sizeof(s_latest_result));
    s_previous_label = POSTURE_CLASS_NONE;
    s_label_count = 0;
    s_none_count = 0;
    s_latched_posture = POSTURE_CLASS_NONE;
}

void Model_Posture_FeedFrame(const VL53LMZ_ResultsData *p_res)
{
    if (p_res == NULL) return;

    uint32_t t_start = DWT->CYCCNT;

    float min_dist = 4000.0f;
    int min_idx = -1;

    // 1. Find closest valid target in 8x8 grid
    for (int i = 0; i < 64; i++) {
        uint8_t status = p_res->target_status[i];
        if (p_res->nb_target_detected[i] > 0 && 
            (status == 5 || status == 9 || status == 6 || status == 10)) {
            float d = (float)p_res->distance_mm[i];
            if (d < min_dist && d > 30.0f) {
                min_dist = d;
                min_idx = i;
            }
        }
    }

    // Validate subject hand distance bounds [100 mm, 400 mm]
    bool valid_frame = (min_dist >= POSTURE_MIN_DISTANCE && min_dist <= POSTURE_MAX_DISTANCE);
    uint8_t current_raw_posture = POSTURE_CLASS_NONE;
    float current_confidence = 0.0f;

    if (valid_frame && min_idx >= 0 && s_ai_ready) {
        // 2. Background subtraction and RobustScaler normalization into (1, 2, 8, 8) planar NCHW tensor
        float in_tensor[128];
        float bg_threshold = min_dist + POSTURE_BG_REMOVAL;

        for (int i = 0; i < 64; i++) {
            uint8_t status = p_res->target_status[i];
            uint8_t nb = p_res->nb_target_detected[i];
            float d_raw = (float)p_res->distance_mm[i];
            float s_raw = (float)p_res->signal_per_spad[i];

            bool zone_valid = (nb > 0) &&
                              (status == 5 || status == 9 || status == 6 || status == 10) &&
                              (d_raw >= 50.0f) &&
                              (d_raw <= bg_threshold);

            float d = zone_valid ? d_raw : DEFAULT_RANGING_VALUE;
            float s = zone_valid ? s_raw : DEFAULT_SIGNAL_VALUE;

            // ST Edge AI expects Planar NCHW (1, 2, 8, 8) [Ch0: Distance 0..63, Ch1: Signal 64..127]
            in_tensor[0 * 64 + i] = (d - NORMALIZATION_RANGING_CENTER) / NORMALIZATION_RANGING_IQR;
            in_tensor[1 * 64 + i] = (s - NORMALIZATION_SIGNAL_CENTER) / NORMALIZATION_SIGNAL_IQR;
        }

        // 3. Execute 2D-CNN inference
        memcpy(s_ai_input[0].data, in_tensor, sizeof(in_tensor));
        ai_i32 nbatch = ai_posture_nn_run(s_posture_network, s_ai_input, s_ai_output);
        if (nbatch > 0) {
            const float *out_logits = (const float *)s_ai_output[0].data;
            float probs[POSTURE_CLASSES_COUNT];
            softmax6(out_logits, probs);

            uint8_t best_cls = POSTURE_CLASS_NONE;
            float best_prob = probs[0];
            for (int c = 1; c < POSTURE_CLASSES_COUNT; c++) {
                if (probs[c] > best_prob) {
                    best_prob = probs[c];
                    best_cls = (uint8_t)c;
                }
            }

            if (best_prob >= POSTURE_CONFIDENCE_THRESH && best_cls != POSTURE_CLASS_NONE) {
                current_raw_posture = best_cls;
                current_confidence = best_prob;
            } else {
                current_raw_posture = POSTURE_CLASS_NONE;
                current_confidence = probs[POSTURE_CLASS_NONE];
            }
        }
    }

    // 4. Temporal Debounce Filter (3-frame latching, anti-flicker)
    bool is_new = false;
    if (current_raw_posture == s_previous_label && current_raw_posture != POSTURE_CLASS_NONE) {
        if (s_label_count < POSTURE_FILTER_N) {
            s_label_count++;
        }
        if (s_label_count == POSTURE_FILTER_N && s_latched_posture != current_raw_posture) {
            s_latched_posture = current_raw_posture;
            is_new = true;
        }
        s_none_count = 0;
    } else {
        s_label_count = 0;
        if (current_raw_posture == POSTURE_CLASS_NONE) {
            if (s_none_count < POSTURE_FILTER_N) {
                s_none_count++;
            } else {
                s_latched_posture = POSTURE_CLASS_NONE;
            }
        } else {
            s_none_count = 0;
        }
    }
    s_previous_label = current_raw_posture;

    uint32_t t_end = DWT->CYCCNT;
    uint32_t cycles = t_end - t_start;
    uint32_t latency_us = cycles / (SystemCoreClock / 1000000);

    s_latest_result.detected_posture = s_latched_posture;
    s_latest_result.confidence = current_confidence;
    s_latest_result.latency_us = latency_us;
    s_latest_result.is_valid_hand = valid_frame;
    s_latest_result.is_new_event = is_new;

    if (is_new && s_latched_posture != POSTURE_CLASS_NONE) {
        printf("[AI POSTURE] >>> DETECTED: %s (Dist: %.0f mm | Conf: %.1f%% | Latency: %lu us) <<<\r\n",
               s_posture_names[s_latched_posture], min_dist, current_confidence * 100.0f, latency_us);
    }
}

const Pipeline_PostureResult_t* Model_Posture_GetResult(void)
{
    return &s_latest_result;
}

const char* Model_Posture_GetClassName(uint8_t cls)
{
    if (cls < POSTURE_CLASSES_COUNT) return s_posture_names[cls];
    return "Unknown";
}
