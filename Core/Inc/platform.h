/*
 * platform.h
 * Platform SPI Driver Header for VL53LMZ ToF on STM32H533
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
  * @file    platform.h
  * @brief   Platform header for ST VL53LMZ ULD SPI implementation on STM32H533
  ******************************************************************************
  */

#ifndef PLATFORM_H_
#define PLATFORM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>
#include "main.h"

/**
 * @brief Structure VL53LMZ_Platform required by the ST ULD API
 */
typedef struct
{
    uint16_t address; /* Target address (required field by API) */
} VL53LMZ_Platform;

/**
 * @brief Number of targets per zone (1 to 4).
 * 1 target per zone minimizes RAM usage and transfer time.
 */
#define VL53LMZ_NB_TARGET_PER_ZONE  1

/* Low-level platform driver functions invoked by ULD */
uint8_t RdByte(VL53LMZ_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_value);
uint8_t WrByte(VL53LMZ_Platform *p_platform, uint16_t RegisterAdress, uint8_t value);
uint8_t RdMulti(VL53LMZ_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_values, uint32_t size);
uint8_t WrMulti(VL53LMZ_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_values, uint32_t size);
void SwapBuffer(uint8_t *buffer, uint16_t size);
uint8_t WaitMs(VL53LMZ_Platform *p_platform, uint32_t TimeMs);

/* Hardware Reset */
void Reset_Sensor(VL53LMZ_Platform *p_platform);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_H_ */
