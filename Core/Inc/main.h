/* USER CODE BEGIN Header */
/*
 * main.h
 * Global system definitions and hardware pin mappings for EdgeSense
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
  * @file           : main.h
  * @brief          : Header for main.c file - Common defines & peripheral interfaces
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h5xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);
void MX_USB_PCD_Init(void);

/* USER CODE BEGIN EFP */
uint8_t Set_Sensor_Profile(uint8_t resolution, uint16_t num_bins, uint8_t frequency_hz, uint8_t sub_sample);
uint8_t Get_Sensor_Resolution(void);
uint16_t Get_Sensor_NumBins(void);
uint8_t Get_Sensor_SubSample(void);
uint8_t Get_Sensor_Frequency(void);
void* Get_CNH_Configuration(void);
/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define VL_NCS_Pin GPIO_PIN_4
#define VL_NCS_GPIO_Port GPIOA
#define VL_INT_Pin GPIO_PIN_0
#define VL_INT_GPIO_Port GPIOB
#define VL_XSHUT_Pin GPIO_PIN_1
#define VL_XSHUT_GPIO_Port GPIOB
#define VL_SYNC_Pin GPIO_PIN_2
#define VL_SYNC_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
