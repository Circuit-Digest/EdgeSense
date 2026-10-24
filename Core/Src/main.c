/* USER CODE BEGIN Header */
/*
 * main.c
 * Main firmware implementation for EdgeSense
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
  * @file           : main.c
  * @brief          : Main program body - EdgeSense System Controller
  *                   Capabilities: 3D Gesture (8x8), Surface Material 1D-CNN (4x4 CNH),
  *                                 Hand Posture 2D-CNN (8x8), Optical Smoke Sensing
  *                   Peripherals : VL53L8CH ToF (SPI1), High-Speed USB CDC, USART1
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "usbd_cdc_if.h"
#include "vl53lmz_api.h"
#include "vl53lmz_plugin_cnh.h"
#include "platform.h"
#include "at_cmd.h"
#include "pipeline_manager.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

SPI_HandleTypeDef hspi1;

UART_HandleTypeDef huart1;

PCD_HandleTypeDef hpcd_USB_DRD_FS;

/* USER CODE BEGIN PV */
extern volatile uint32_t g_usb_irq_count;

static VL53LMZ_Configuration         g_vl53lmz_dev;
static VL53LMZ_ResultsData           g_vl53lmz_results;
static VL53LMZ_Motion_Configuration  g_cnh_config;
static cnh_data_buffer_t             g_cnh_data_buffer;
static uint32_t                      g_cnh_data_size = 0;
static int16_t                       g_cnh_min_dist = 0;
static int16_t                       g_cnh_max_dist = 0;
static uint8_t  g_sensor_initialized = 0;
uint8_t         g_ranging_active     = 0;
static uint32_t g_frame_count        = 0;
static uint32_t g_last_frame_tick    = 0;
float           g_fps                = 0.0f;
static char     s_json_buf[6144];
static uint8_t  g_current_resolution = 64;
static uint16_t g_cnh_num_bins        = 16;
static uint8_t  g_cnh_sub_sample      = 4;
static uint8_t  g_ranging_freq_hz     = 15;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_ICACHE_Init(void);
static void MX_SPI1_Init(void);
static void MX_USART1_UART_Init(void);
/* USER CODE BEGIN PFP */
void MX_USB_PCD_Init(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// Redirect printf to USB Type-C Virtual COM Port (Non-blocking high-speed)
int __io_putchar(int ch) {
    uint8_t c = (uint8_t)ch;
    CDC_Transmit_FS(&c, 1);
    return ch;
}

// Low-level SPI write for VL53L8CH
void VL_WriteByte(uint16_t reg, uint8_t data) {
    uint8_t tx[3];
    // In VL53L8CH SPI: write bit (bit 15) is 1
    tx[0] = (uint8_t)((reg >> 8) | 0x80);
    tx[1] = (uint8_t)(reg & 0xFF);
    tx[2] = data;

    HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_RESET); // Chip Select LOW
    HAL_SPI_Transmit(&hspi1, tx, 3, 100);
    HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_SET);   // Chip Select HIGH
}

// Low-level SPI read for VL53L8CH
uint8_t VL_ReadByte(uint16_t reg) {
    uint8_t tx[2];
    uint8_t rx = 0;
    // In VL53L8CH SPI: read bit (bit 15) is 0
    tx[0] = (uint8_t)((reg >> 8) & 0x7F);
    tx[1] = (uint8_t)(reg & 0xFF);

    HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_RESET); // Chip Select LOW
    HAL_SPI_Transmit(&hspi1, tx, 2, 100);                             // Send 16-bit address
    HAL_SPI_Receive(&hspi1, &rx, 1, 100);                             // Read byte
    HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_SET);   // Chip Select HIGH
    return rx;
}



// Ultra-fast integer-to-string append (avoids printf overhead)
static inline char* append_int(char *p, int val) {
    if (val < 0) {
        *p++ = '-';
        val = -val;
    }
    char tmp[10];
    int i = 0;
    do {
        tmp[i++] = (char)('0' + (val % 10));
        val /= 10;
    } while (val > 0);
    while (i > 0) {
        *p++ = tmp[--i];
    }
    return p;
}

// Ultra-fast 1-decimal-place float append (e.g. 12.3)
static inline char* append_fixed1(char *p, float val) {
    if (val < 0.0f) {
        *p++ = '-';
        val = -val;
    }
    uint32_t vi = (uint32_t)val;
    uint32_t vf = (uint32_t)((val - (float)vi) * 10.0f + 0.5f);
    if (vf >= 10) {
        vi++;
        vf = 0;
    }
    char tmp[10];
    int i = 0;
    do {
        tmp[i++] = (char)('0' + (vi % 10));
        vi /= 10;
    } while (vi > 0);
    while (i > 0) {
        *p++ = tmp[--i];
    }
    *p++ = '.';
    *p++ = (char)('0' + vf);
    return p;
}

uint8_t Get_Sensor_Resolution(void) { return g_current_resolution; }
uint16_t Get_Sensor_NumBins(void) { return g_cnh_num_bins; }
uint8_t Get_Sensor_SubSample(void) { return g_cnh_sub_sample; }
uint8_t Get_Sensor_Frequency(void) { return g_ranging_freq_hz; }
void* Get_CNH_Configuration(void) { return (void*)&g_cnh_config; }

/// High-Speed Atomic JSON Packet Streaming (Dynamic Resolution & Bins)
static void Print_JSON_Stream(VL53LMZ_ResultsData *p_res, uint32_t frame_no) {
    char *p = s_json_buf;
    uint8_t num_zones = g_current_resolution;
    uint16_t nb_bins = g_cnh_num_bins;

    // 1. Frame metadata
    *p++ = '{';
    *p++ = '"'; *p++ = 'f'; *p++ = '"'; *p++ = ':';
    p = append_int(p, (int)frame_no);
    *p++ = ',';
    *p++ = '"'; *p++ = 't'; *p++ = '"'; *p++ = ':';
    p = append_int(p, (int)p_res->silicon_temp_degc);
    *p++ = ',';
    *p++ = '"'; *p++ = 'r'; *p++ = 'e'; *p++ = 's'; *p++ = '"'; *p++ = ':';
    p = append_int(p, (int)num_zones);
    *p++ = ',';
    *p++ = '"'; *p++ = 'b'; *p++ = 'i'; *p++ = 'n'; *p++ = 's'; *p++ = '"'; *p++ = ':';
    p = append_int(p, (int)nb_bins);

    // 2. Distance values (16 or 64)
    *p++ = ',';
    *p++ = '"'; *p++ = 'd'; *p++ = '"'; *p++ = ':'; *p++ = '[';
    for (int i = 0; i < num_zones; i++) {
        if (i > 0) *p++ = ',';
        uint8_t status = p_res->target_status[i];
        int16_t d = (status == 5 || status == 6 || status == 9 || status == 4) ? p_res->distance_mm[i] : -1;
        p = append_int(p, (int)d);
    }
    *p++ = ']';

    // 3. Peak Signal values (kcps/SPAD)
    *p++ = ',';
    *p++ = '"'; *p++ = 's'; *p++ = '"'; *p++ = ':'; *p++ = '[';
    for (int i = 0; i < num_zones; i++) {
        if (i > 0) *p++ = ',';
        uint32_t sig = p_res->signal_per_spad[i];
        p = append_int(p, (int)sig);
    }
    *p++ = ']';

    // 4. Native Ambient Noise values (kcps/spad)
    *p++ = ',';
    *p++ = '"'; *p++ = 'a'; *p++ = '"'; *p++ = ':'; *p++ = '[';
    for (int i = 0; i < num_zones; i++) {
        if (i > 0) *p++ = ',';
        p = append_int(p, (int)p_res->ambient_per_spad[i]);
    }
    *p++ = ']';

    // 4. Zone Histograms Concurrent Matrix
    *p++ = ',';
    *p++ = '"'; *p++ = 'h'; *p++ = '"'; *p++ = ':'; *p++ = '[';
    for (int z = 0; z < num_zones; z++) {
        if (z > 0) *p++ = ',';
        *p++ = '[';

        int32_t *p_hist = NULL;
        int8_t *p_hist_scaler = NULL;
        int32_t *p_ambient = NULL;
        int8_t *p_ambient_scaler = NULL;

        vl53lmz_cnh_get_block_addresses(&g_cnh_config, z, g_cnh_data_buffer,
                                        &p_hist, &p_hist_scaler, &p_ambient, &p_ambient_scaler);

        if (p_hist && p_hist_scaler) {
            for (int b = 0; b < nb_bins; b++) {
                if (b > 0) *p++ = ',';
                float val = ((float)p_hist[b]) / (float)(1 << p_hist_scaler[b]);
                p = append_fixed1(p, val);
            }
        } else {
            for (int b = 0; b < nb_bins; b++) {
                if (b > 0) *p++ = ',';
                *p++ = '0';
            }
        }
        *p++ = ']';
    }
    *p++ = ']';
    *p++ = '}';
    *p++ = '\r';
    *p++ = '\n';
    *p = '\0';

    int len = p - s_json_buf;

    // Single Atomic Non-blocking USB CDC Transmit
    if (len > 0 && len < (int)sizeof(s_json_buf)) {
        CDC_Transmit_FS((uint8_t*)s_json_buf, (uint16_t)len);
    }
}

uint8_t Init_VL53LMZ_Sensor(void);

uint8_t Set_Sensor_Profile(uint8_t resolution, uint16_t num_bins, uint8_t frequency_hz, uint8_t sub_sample) {
    if (!g_sensor_initialized) {
        return VL53LMZ_STATUS_ERROR;
    }
    if (resolution != 16 && resolution != 64) {
        return VL53LMZ_STATUS_INVALID_PARAM;
    }

    uint8_t status = VL53LMZ_STATUS_OK;
    uint8_t was_ranging = g_ranging_active;

    printf("\r\n[PROFILE] Reconfiguring sensor to %s (%d Zones, %d Bins, SubSample %d @ %d Hz)...\r\n",
           (resolution == 16) ? "4x4" : "8x8", resolution, num_bins, sub_sample, frequency_hz);

    // 1. Stop sensor ranging if active
    if (g_ranging_active) {
        status = vl53lmz_stop_ranging(&g_vl53lmz_dev);
        g_ranging_active = 0;
        if (status != VL53LMZ_STATUS_OK) {
            printf(">>> [FAIL] vl53lmz_stop_ranging failed! status=0x%02X <<<\r\n", status);
            return status;
        }
    }

    // 2. Set resolution, frequency, and integration time
    status = vl53lmz_set_resolution(&g_vl53lmz_dev, resolution);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_set_resolution failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    status = vl53lmz_set_ranging_frequency_hz(&g_vl53lmz_dev, frequency_hz);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_set_ranging_frequency_hz failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    uint32_t integration_time = (resolution == 16) ? 15 : 20;
    status = vl53lmz_set_integration_time_ms(&g_vl53lmz_dev, integration_time);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_set_integration_time_ms failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    status = vl53lmz_set_ranging_mode(&g_vl53lmz_dev, VL53LMZ_RANGING_MODE_CONTINUOUS);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_set_ranging_mode failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 3. Configure CNH Plugin
    status = vl53lmz_cnh_init_config(&g_cnh_config, 0, (int16_t)num_bins, (int16_t)sub_sample);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_init_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    int16_t grid_dim = (resolution == 16) ? 4 : 8;
    status = vl53lmz_cnh_create_agg_map(&g_cnh_config, (int16_t)resolution, 0, 0, 1, 1, grid_dim, grid_dim);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_create_agg_map failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    status = vl53lmz_cnh_calc_required_memory(&g_cnh_config, &g_cnh_data_size);
    if (status != VL53LMZ_STATUS_OK || g_cnh_data_size > VL53LMZ_CNH_MAX_DATA_BYTES) {
        printf(">>> [FAIL] CNH memory calculation error: %lu bytes <<<\r\n", g_cnh_data_size);
        return VL53LMZ_STATUS_ERROR;
    }
    vl53lmz_cnh_calc_min_max_distance(&g_cnh_config, &g_cnh_min_dist, &g_cnh_max_dist);
    printf("   CNH Memory: %lu bytes (Max: %lu) | Distance Span: %d - %d mm\r\n",
           g_cnh_data_size, VL53LMZ_CNH_MAX_DATA_BYTES, g_cnh_min_dist, g_cnh_max_dist);

    status = vl53lmz_cnh_send_config(&g_vl53lmz_dev, &g_cnh_config);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_send_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 4. Build output descriptor with CNH block
    status = vl53lmz_create_output_config(&g_vl53lmz_dev);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_create_output_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    union Block_header cnh_data_bh;
    cnh_data_bh.idx = VL53LMZ_CNH_DATA_IDX;
    cnh_data_bh.type = 4;
    cnh_data_bh.size = g_cnh_data_size / 4;
    status = vl53lmz_add_output_block(&g_vl53lmz_dev, cnh_data_bh.bytes);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_add_output_block failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 5. Update global profile state
    g_current_resolution = resolution;
    g_cnh_num_bins = num_bins;
    g_cnh_sub_sample = sub_sample;
    g_ranging_freq_hz = frequency_hz;

    // 6. Resume ranging if it was active previously
    if (was_ranging) {
        status = vl53lmz_send_output_config_and_start(&g_vl53lmz_dev);
        if (status == VL53LMZ_STATUS_OK) {
            g_ranging_active = 1;
            printf("[PROFILE] Ranging restarted successfully.\r\n");
        } else {
            printf(">>> [FAIL] Failed to restart ranging! status=0x%02X <<<\r\n", status);
            return status;
        }
    }

    printf("[PROFILE] Reconfiguration complete: %s (%d Zones, %d Bins @ %d Hz)\r\n",
           (resolution == 16) ? "4x4" : "8x8", resolution, num_bins, frequency_hz);
    return VL53LMZ_STATUS_OK;
}

void Set_Sensor_Ranging(uint8_t enable) {
    if (enable && !g_ranging_active) {
        uint8_t status = vl53lmz_send_output_config_and_start(&g_vl53lmz_dev);
        if (status == VL53LMZ_STATUS_OK) {
            g_ranging_active = 1;
            printf("[SENSOR] Ranging resumed.\r\n");
        } else {
            printf("[SENSOR] Resume failed (0x%02X), reinitializing...\r\n", status);
            Init_VL53LMZ_Sensor();
        }
    } else if (!enable && g_ranging_active) {
        uint8_t status = vl53lmz_stop_ranging(&g_vl53lmz_dev);
        g_ranging_active = 0;
        printf("[SENSOR] Ranging paused.\r\n");
        (void)status;
    }
}

// Full initialization sequence using ST VL53LMZ ULD driver with CNH plugin
uint8_t Init_VL53LMZ_Sensor(void) {
    uint8_t status = 0;
    uint8_t is_alive = 0;

    printf("\r\n==============================================\r\n");
    printf("--- Initializing VL53L8CH ULD Driver with CNH ---\r\n");
    printf("SPI1 Clock: 3.000 MHz (PLL1Q 48.0 MHz / 16)\r\n");

    // 1. Hardware Reset via XSHUT pin
    printf("1. Hardware Reset via VL_XSHUT (PB1)...\r\n");
    Reset_Sensor(&(g_vl53lmz_dev.platform));

    // 2. Check if sensor silicon is alive on SPI
    printf("2. Probing sensor on SPI1...\r\n");
    status = vl53lmz_is_alive(&g_vl53lmz_dev, &is_alive);
    if (!is_alive || status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] Sensor not responding on SPI! status=0x%02X, is_alive=%d <<<\r\n", status, is_alive);
        return 1;
    }
    printf("   [OK] Sensor alive and responding!\r\n");

    // 3. Upload firmware to sensor RAM (~80 KB payload)
    printf("3. Uploading firmware to sensor internal RAM (~80 KB via SPI)...\r\n");
    uint32_t t_start = HAL_GetTick();
    status = vl53lmz_init(&g_vl53lmz_dev);
    uint32_t t_elapsed = HAL_GetTick() - t_start;

    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_init failed! status=0x%02X <<<\r\n", status);
        return status;
    }
    printf("   >>> [SUCCESS] VL53L8CH ULD Ready! (API Rev: %s, took %lu ms) <<<\r\n", 
           VL53LMZ_API_REVISION, t_elapsed);

    // 4. Configure Ranging parameters for 8x8 Grid @ 15 Hz
    printf("4. Configuring 8x8 Grid (64 Zones) @ 15 Hz Continuous Ranging...\r\n");
    status |= vl53lmz_set_resolution(&g_vl53lmz_dev, 64);
    status |= vl53lmz_set_ranging_frequency_hz(&g_vl53lmz_dev, 15);
    status |= vl53lmz_set_integration_time_ms(&g_vl53lmz_dev, 20);
    status |= vl53lmz_set_ranging_mode(&g_vl53lmz_dev, VL53LMZ_RANGING_MODE_CONTINUOUS);

    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] Basic ranging configuration failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 5. Configure CNH (Compact Normalized Histogram - 64 Zones x 16 Bins)
    printf("5. Configuring CNH (64 Zones x 16 Bins, SubSample 4)...\r\n");
    status = vl53lmz_cnh_init_config(&g_cnh_config, 0, 16, 4);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_init_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    status = vl53lmz_cnh_create_agg_map(&g_cnh_config, 64, 0, 0, 1, 1, 8, 8);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_create_agg_map failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    status = vl53lmz_cnh_calc_required_memory(&g_cnh_config, &g_cnh_data_size);
    if (status != VL53LMZ_STATUS_OK || g_cnh_data_size > VL53LMZ_CNH_MAX_DATA_BYTES) {
        printf(">>> [FAIL] CNH memory calculation error: %lu bytes <<<\r\n", g_cnh_data_size);
        return status;
    }
    vl53lmz_cnh_calc_min_max_distance(&g_cnh_config, &g_cnh_min_dist, &g_cnh_max_dist);
    printf("   CNH Memory: %lu bytes (Max: %lu) | Distance Span: %d - %d mm\r\n",
           g_cnh_data_size, VL53LMZ_CNH_MAX_DATA_BYTES, g_cnh_min_dist, g_cnh_max_dist);

    status = vl53lmz_cnh_send_config(&g_vl53lmz_dev, &g_cnh_config);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_cnh_send_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 6. Build and send output configuration containing CNH block
    printf("6. Building output descriptor with CNH block 0x%04X...\r\n", VL53LMZ_CNH_DATA_IDX);
    status = vl53lmz_create_output_config(&g_vl53lmz_dev);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_create_output_config failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    union Block_header cnh_data_bh;
    cnh_data_bh.idx = VL53LMZ_CNH_DATA_IDX;
    cnh_data_bh.type = 4;
    cnh_data_bh.size = g_cnh_data_size / 4;
    status = vl53lmz_add_output_block(&g_vl53lmz_dev, cnh_data_bh.bytes);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_add_output_block failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    // 7. Start ranging with CNH output
    printf("7. Starting sensor ranging with CNH stream...\r\n");
    status = vl53lmz_send_output_config_and_start(&g_vl53lmz_dev);
    if (status != VL53LMZ_STATUS_OK) {
        printf(">>> [FAIL] vl53lmz_send_output_config_and_start failed! status=0x%02X <<<\r\n", status);
        return status;
    }

    g_sensor_initialized = 1;
    g_ranging_active = 1;
    printf(">>> [SUCCESS] VL53L8CH Sensor Active! <<<\r\n");
    printf("Control Plane: Send 'AT+HELP' for available AT commands\r\n");
    printf("==============================================\r\n");
    return 0;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ICACHE_Init();
  MX_SPI1_Init();
  MX_USART1_UART_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
  printf("\r\n==============================================\r\n");
  printf("EdgeSense STM32H533 Initializing...\r\n");
  printf("Core Clock (SYSCLK): %lu MHz\r\n", HAL_RCC_GetSysClockFreq() / 1000000);

  // 1. Initialize USB PCD Hardware & Classic ST USB Device Stack
  printf("Starting USB Controller & Classic ST USB Device CDC Stack...\r\n");
  MX_USB_PCD_Init();

  // 2. Connect internal D+ pull-up to host PC
  printf("Connecting USB D+ Pull-Up to PC Host...\r\n");
  HAL_PCD_DevDisconnect(&hpcd_USB_DRD_FS);
  HAL_Delay(100);
  HAL_PCD_DevConnect(&hpcd_USB_DRD_FS);
  HAL_StatusTypeDef pcd_status = HAL_PCD_Start(&hpcd_USB_DRD_FS);
  printf("USB Transceiver Started: %s\r\n", (pcd_status == HAL_OK) ? "SUCCESS" : "FAILED");
  printf("Communication Active: Hardware UART (J5) + USB-C (J4)\r\n");
  printf("==============================================\r\n");

  // 3. Initialize VL53L8CH ULD Driver with CNH and Start Ranging
  Init_VL53LMZ_Sensor();

  // 4. Initialize Asynchronous AT Command Processor
  AT_Cmd_Init();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint32_t last_paused_tick = HAL_GetTick();
  uint8_t rx_char = 0;

  while (1)
  {
    /* USER CODE END WHILE */

  MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
    // 1. Pump non-blocking USB CDC output
    USB_CDC_TxPump();

    // 2. Process UART keyboard input
    if (HAL_UART_Receive(&huart1, &rx_char, 1, 0) == HAL_OK) {
        AT_Cmd_FeedChar((char)rx_char);
    }

    // 3. Process pending AT commands in main thread context (non-blocking)
    AT_Cmd_Process();

    // 4. Sensor data acquisition when VL_INT is LOW (Active-Low Interrupt: Data Ready)
    if (g_ranging_active && HAL_GPIO_ReadPin(VL_INT_GPIO_Port, VL_INT_Pin) == GPIO_PIN_RESET) {
        uint8_t status = vl53lmz_get_ranging_data(&g_vl53lmz_dev, &g_vl53lmz_results);
        if (status == VL53LMZ_STATUS_OK) {
            // Extract the CNH raw histogram block from the transfer buffer
            status = vl53lmz_results_extract_block(&g_vl53lmz_dev, VL53LMZ_CNH_DATA_IDX, 
                                                  (uint8_t *)g_cnh_data_buffer, (uint16_t)g_cnh_data_size);
            if (status == VL53LMZ_STATUS_OK) {
                g_frame_count++;
                uint32_t now = HAL_GetTick();
                if (now != g_last_frame_tick) {
                    g_fps = 1000.0f / (float)(now - g_last_frame_tick);
                }
                g_last_frame_tick = now;

                // Streaming telemetry according to active stream mode
                StreamMode_t sm = Pipeline_GetStreamMode();
                if (sm == STREAM_MODE_RAW || sm == STREAM_MODE_DEBUG) {
                    Print_JSON_Stream(&g_vl53lmz_results, g_frame_count);
                }

                // Real-Time Multi-Capability AI Pipeline (Gesture, Posture, Surface, Smoke)
                Pipeline_FeedFrame(&g_vl53lmz_results, (const uint32_t *)g_cnh_data_buffer, g_cnh_data_size);
            }
        }
    }

    // 5. Heartbeat when paused
    if (!g_ranging_active && (HAL_GetTick() - last_paused_tick >= 5000)) {
        last_paused_tick = HAL_GetTick();
        printf("[PAUSED] Send 'AT+RESUME' to start ranging, 'AT+HELP' for command list.\r\n");
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_CRSInitTypeDef RCC_CRSInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSI
                              |RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV2;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLL1_SOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1_VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1_VCORANGE_WIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable the CRS APB clock
  */
  __HAL_RCC_CRS_CLK_ENABLE();

  /** Configures CRS
  */
  RCC_CRSInitStruct.Prescaler = RCC_CRS_SYNC_DIV1;
  RCC_CRSInitStruct.Source = RCC_CRS_SYNC_SOURCE_USB;
  RCC_CRSInitStruct.Polarity = RCC_CRS_SYNC_POLARITY_RISING;
  RCC_CRSInitStruct.ReloadValue = __HAL_RCC_CRS_RELOADVALUE_CALCULATE(48000000,1000);
  RCC_CRSInitStruct.ErrorLimitValue = 34;
  RCC_CRSInitStruct.HSI48CalibrationValue = 32;

  HAL_RCCEx_CRSConfig(&RCC_CRSInitStruct);

  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();

  /** Configure the programming delay
  */
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_2);
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_CKPER;
  PeriphClkInitStruct.CkperClockSelection = RCC_CLKPSOURCE_HSI;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache in 1-way (direct mapped cache)
  */
  if (HAL_ICACHE_ConfigAssociativityMode(ICACHE_1WAY) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi1.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 0x7;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
  hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  hspi1.Init.ReadyMasterManagement = SPI_RDY_MASTER_MANAGEMENT_INTERNALLY;
  hspi1.Init.ReadyPolarity = SPI_RDY_POLARITY_HIGH;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USB Initialization Function
  * @param None
  * @retval None
  */
void MX_USB_PCD_Init(void)
{

  /* USER CODE BEGIN USB_Init 0 */

  /* USER CODE END USB_Init 0 */

  /* USER CODE BEGIN USB_Init 1 */

  /* USER CODE END USB_Init 1 */
  hpcd_USB_DRD_FS.Instance = USB_DRD_FS;
  hpcd_USB_DRD_FS.Init.dev_endpoints = 8;
  hpcd_USB_DRD_FS.Init.speed = USBD_FS_SPEED;
  hpcd_USB_DRD_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_DRD_FS.Init.Sof_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.battery_charging_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.vbus_sensing_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.bulk_doublebuffer_enable = DISABLE;
  hpcd_USB_DRD_FS.Init.iso_singlebuffer_enable = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_DRD_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_Init 2 */
  MX_USB_Device_Init();
  /* USER CODE END USB_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, VL_XSHUT_Pin|VL_SYNC_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : VL_NCS_Pin */
  GPIO_InitStruct.Pin = VL_NCS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(VL_NCS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : VL_INT_Pin */
  GPIO_InitStruct.Pin = VL_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(VL_INT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : VL_XSHUT_Pin VL_SYNC_Pin */
  GPIO_InitStruct.Pin = VL_XSHUT_Pin|VL_SYNC_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};
  MPU_Attributes_InitTypeDef MPU_AttributesInit = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region 0 and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x08FFF000;
  MPU_InitStruct.LimitAddress = 0x08FFFFFF;
  MPU_InitStruct.AttributesIndex = MPU_ATTRIBUTES_NUMBER0;
  MPU_InitStruct.AccessPermission = MPU_REGION_ALL_RO;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  /** Initializes and configures the Attribute 0 and the memory to be protected
  */
  MPU_AttributesInit.Number = MPU_ATTRIBUTES_NUMBER0;
  MPU_AttributesInit.Attributes = INNER_OUTER(MPU_NOT_CACHEABLE);

  HAL_MPU_ConfigMemoryAttributes(&MPU_AttributesInit);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param None
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
