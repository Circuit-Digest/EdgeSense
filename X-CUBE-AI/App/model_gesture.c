/*
 * model_gesture.c
 * 3D Dynamic Gesture Recognition Implementation (15-frame window)
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
  * @file    model_gesture.c
  * @brief   3D Dynamic Gesture Recognition Implementation
  *          Features: 15-frame temporal window @ 15 Hz on 8x8 ToF grid,
  *                    Spatial-temporal feature extraction & MLP/CNN inference,
  *                    Classes: IDLE, SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN
  ******************************************************************************
  */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "main.h"
#include "model_gesture.h"
#include "gesture_nn.h"
#include "gesture_nn_data.h"

static ai_handle s_network = AI_HANDLE_NULL;
static ai_buffer *s_ai_input = NULL;
static ai_buffer *s_ai_output = NULL;
static bool s_ai_ready = false;

// 15-Frame Rolling Window of 64 Zones (-1 for empty/invalid)
static int16_t s_window[GESTURE_WINDOW_SIZE][64];
static uint8_t s_window_count = 0;

// Planar tensor input buffer (15 frames * 8 rows * 8 cols = 960 floats)
static float s_input_tensor[GESTURE_INPUT_DIM];
static float s_out_probs[GESTURE_CLASSES_COUNT];

static Pipeline_GestureResult_t s_latest_result = {0};
static uint32_t s_cooldown_until_tick = 0;

static const char *s_class_names[GESTURE_CLASSES_COUNT] = {
    "IDLE", "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"
};

static void softmax5(const float *logits, float *probs)
{
    float max_l = logits[0];
    for (int i = 1; i < GESTURE_CLASSES_COUNT; i++) {
        if (logits[i] > max_l) {
            max_l = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < GESTURE_CLASSES_COUNT; i++) {
        probs[i] = expf(logits[i] - max_l);
        sum += probs[i];
    }
    float inv_sum = (sum > 1e-6f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < GESTURE_CLASSES_COUNT; i++) {
        probs[i] *= inv_sum;
    }
}

bool Model_Gesture_Init(void)
{
    // Enable Cortex-M33 DWT cycle counter for microsecond benchmark
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    ai_handle act_addr[] = { Pipeline_GetSharedActivationPool() };
    ai_error err = ai_gesture_nn_create_and_init(&s_network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("[AI-ERR] ai_gesture_nn_create_and_init failed: %d\r\n", err.type);
        s_ai_ready = false;
        return false;
    }

    ai_u16 n_in = 0, n_out = 0;
    s_ai_input = ai_gesture_nn_inputs_get(s_network, &n_in);
    s_ai_output = ai_gesture_nn_outputs_get(s_network, &n_out);
    s_ai_ready = true;
    Model_Gesture_Reset();
    printf("[AI OK] Model 'gesture_nn' (Spatio-Temporal 2D-CNN) loaded successfully.\r\n");
    return true;
}

void Model_Gesture_Reset(void)
{
    s_window_count = 0;
    for (int t = 0; t < GESTURE_WINDOW_SIZE; t++) {
        for (int z = 0; z < 64; z++) {
            s_window[t][z] = -1; // Background empty distance
        }
    }
    memset(&s_latest_result, 0, sizeof(s_latest_result));
    s_cooldown_until_tick = 0;
}

void Model_Gesture_FeedFrame(const VL53LMZ_ResultsData *p_res)
{
    if (!s_ai_ready || p_res == NULL) return;

    // 1. Shift rolling window and insert new sample
    memmove(&s_window[0][0], &s_window[1][0], (GESTURE_WINDOW_SIZE - 1) * 64 * sizeof(int16_t));
    for (int i = 0; i < 64; i++) {
        uint8_t status = p_res->target_status[i];
        int16_t d = (status == 5 || status == 6 || status == 9 || status == 4) ? (int16_t)p_res->distance_mm[i] : -1;
        s_window[GESTURE_WINDOW_SIZE - 1][i] = d;
    }
    if (s_window_count < GESTURE_WINDOW_SIZE) {
        s_window_count++;
        return; // Prime initial 15-frame history before running live inference
    }

    s_latest_result.is_new_event = false;

    // 2. Normalize 15-frame window into planar (1, 15, 8, 8) tensor
    // Background / empty / invalid: strictly 0.0f
    // Valid hand presence: (400 - d) / 350.0f
    for (int t = 0; t < GESTURE_WINDOW_SIZE; t++) {
        for (int i = 0; i < 64; i++) {
            int16_t d = s_window[t][i];
            float norm = (d >= 50 && d <= 400) ? ((400.0f - (float)d) / 350.0f) : 0.0f;
            s_input_tensor[t * 64 + i] = norm;
        }
    }

    // 3. Setup Cube.AI input & output tensors
    if (s_ai_input[0].data != NULL) {
        memcpy(s_ai_input[0].data, s_input_tensor, GESTURE_INPUT_DIM * sizeof(float));
    } else {
        s_ai_input[0].data = AI_HANDLE_PTR(s_input_tensor);
    }
    if (s_ai_output[0].data == NULL) {
        s_ai_output[0].data = AI_HANDLE_PTR(s_out_probs);
    }

    // 4. Execute Spatio-Temporal 2D-CNN with DWT cycle benchmark
    uint32_t start_cycles = DWT->CYCCNT;
    ai_i32 batch = ai_gesture_nn_run(s_network, s_ai_input, s_ai_output);
    uint32_t end_cycles = DWT->CYCCNT;

    if (batch <= 0) return;

    uint32_t elapsed_cycles = end_cycles - start_cycles;
    uint32_t latency_us = (uint32_t)((uint64_t)elapsed_cycles * 1000000ULL / SystemCoreClock);

    // 5. Apply Softmax normalization to raw logits
    float *logits = (float *)s_ai_output[0].data;
    float probs[GESTURE_CLASSES_COUNT];
    softmax5(logits, probs);

    int best_class = 0;
    float best_prob = probs[0];
    for (int i = 1; i < GESTURE_CLASSES_COUNT; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_class = i;
        }
    }

    // Temporal Motion Energy Guard:
    // A dynamic swipe requires spatial displacement across the 15-frame window.
    // When a hand rests stationary (e.g. after completing a downward swipe),
    // temporal motion energy drops (< 16.0f), immediately returning output to IDLE.
    float motion_energy = 0.0f;
    for (int t = 1; t < GESTURE_WINDOW_SIZE; t++) {
        for (int i = 0; i < 64; i++) {
            float diff = fabsf(s_input_tensor[t * 64 + i] - s_input_tensor[(t - 1) * 64 + i]);
            motion_energy += diff;
        }
    }

    if (motion_energy < 16.0f) {
        best_class = GESTURE_CLASS_IDLE;
        best_prob = 1.0f;
    }

    s_latest_result.detected_class = (uint8_t)best_class;
    s_latest_result.confidence = best_prob;
    s_latest_result.latency_us = latency_us;
    for (int i = 0; i < GESTURE_CLASSES_COUNT; i++) {
        s_latest_result.probabilities[i] = (best_class == GESTURE_CLASS_IDLE && motion_energy < 16.0f) ?
                                           ((i == 0) ? 1.0f : 0.0f) : probs[i];
    }

    // 6. Sliding Window Gesture Debouncing (exact parity with Python Studio)
    uint32_t now_tick = HAL_GetTick();

    if ((int32_t)(now_tick - s_cooldown_until_tick) < 0) {
        return; // During 850 ms cooldown, reject hand retraction/rebound transients
    }

    if (best_class != GESTURE_CLASS_IDLE && best_prob >= 0.80f) {
        s_latest_result.is_new_event = true;
        s_cooldown_until_tick = now_tick + 850; // 850 ms cooldown matches Studio

        const char *name = Model_Gesture_GetClassName((uint8_t)best_class);
        printf("[AI GESTURE] >>> DETECTED: %s (Confidence: %5.1f%% | Latency: %lu us | %lu cycles) <<<\r\n",
               name, (double)(best_prob * 100.0f), (unsigned long)latency_us, (unsigned long)elapsed_cycles);
    }
}

const Pipeline_GestureResult_t* Model_Gesture_GetResult(void)
{
    return &s_latest_result;
}

const char* Model_Gesture_GetClassName(uint8_t cls)
{
    if (cls < GESTURE_CLASSES_COUNT) return s_class_names[cls];
    return "UNKNOWN";
}

GestureFsmState_t Model_Gesture_GetFsmState(void)
{
    return GESTURE_FSM_IDLE;
}
