/*
 * platform.c
 * Platform SPI Driver Implementation for VL53LMZ ToF on STM32H533
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
  * @file    platform.c
  * @brief   Platform SPI driver implementation for ST VL53LMZ ULD on STM32H533
  ******************************************************************************
  */

#include "platform.h"
#include "main.h"

#define VL53LMZ_COMMS_CHUNK_SIZE 1024
#define SPI_WRITE_MASK(x) ((uint16_t)((x) | 0x8000))
#define SPI_READ_MASK(x)  ((uint16_t)((x) & ~0x8000))

extern SPI_HandleTypeDef hspi1;

// Static buffer to avoid allocating large arrays on the stack
static uint8_t s_spi_tx_buf[VL53LMZ_COMMS_CHUNK_SIZE + 2];

uint8_t RdByte(
    VL53LMZ_Platform *p_platform,
    uint16_t RegisterAdress,
    uint8_t *p_value)
{
    return RdMulti(p_platform, RegisterAdress, p_value, 1);
}

uint8_t WrByte(
    VL53LMZ_Platform *p_platform,
    uint16_t RegisterAdress,
    uint8_t value)
{
    return WrMulti(p_platform, RegisterAdress, &value, 1);
}

uint8_t WrMulti(
    VL53LMZ_Platform *p_platform,
    uint16_t RegisterAdress,
    uint8_t *p_values,
    uint32_t size)
{
    (void)p_platform;
    uint8_t status = 0;
    uint32_t position = 0;
    uint32_t data_size = 0;
    uint16_t temp;

    for (position = 0; position < size; position += VL53LMZ_COMMS_CHUNK_SIZE)
    {
        if ((position + VL53LMZ_COMMS_CHUNK_SIZE) > size) {
            data_size = size - position;
        } else {
            data_size = VL53LMZ_COMMS_CHUNK_SIZE;
        }

        temp = RegisterAdress + position;
        s_spi_tx_buf[0] = (uint8_t)(SPI_WRITE_MASK(temp) >> 8);
        s_spi_tx_buf[1] = (uint8_t)(SPI_WRITE_MASK(temp) & 0xFF);

        memcpy(&s_spi_tx_buf[2], &p_values[position], data_size);

        HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_RESET);
        if (HAL_SPI_Transmit(&hspi1, s_spi_tx_buf, (uint16_t)(data_size + 2), 1000) != HAL_OK) {
            status = 1;
        }
        HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_SET);

        if (status != 0) break;
    }

    return status;
}

uint8_t RdMulti(
    VL53LMZ_Platform *p_platform,
    uint16_t RegisterAdress,
    uint8_t *p_values,
    uint32_t size)
{
    (void)p_platform;
    uint8_t status = 0;
    uint32_t position = 0;
    uint32_t data_size = 0;
    uint16_t temp;
    uint8_t addr_header[2];

    for (position = 0; position < size; position += VL53LMZ_COMMS_CHUNK_SIZE)
    {
        if ((position + VL53LMZ_COMMS_CHUNK_SIZE) > size) {
            data_size = size - position;
        } else {
            data_size = VL53LMZ_COMMS_CHUNK_SIZE;
        }

        temp = RegisterAdress + position;
        addr_header[0] = (uint8_t)(SPI_READ_MASK(temp) >> 8);
        addr_header[1] = (uint8_t)(SPI_READ_MASK(temp) & 0xFF);

        HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_RESET);
        if (HAL_SPI_Transmit(&hspi1, addr_header, 2, 100) != HAL_OK) {
            status = 1;
        } else {
            if (HAL_SPI_Receive(&hspi1, &p_values[position], (uint16_t)data_size, 1000) != HAL_OK) {
                status = 1;
            }
        }
        HAL_GPIO_WritePin(VL_NCS_GPIO_Port, VL_NCS_Pin, GPIO_PIN_SET);

        if (status != 0) break;
    }

    return status;
}

void SwapBuffer(
    uint8_t *buffer,
    uint16_t size)
{
    uint32_t i, tmp;

    for (i = 0; i < size; i += 4)
    {
        tmp = ((uint32_t)buffer[i]     << 24)
            | ((uint32_t)buffer[i + 1] << 16)
            | ((uint32_t)buffer[i + 2] << 8)
            | ((uint32_t)buffer[i + 3]);

        memcpy(&buffer[i], &tmp, 4);
    }
}

uint8_t WaitMs(
    VL53LMZ_Platform *p_platform,
    uint32_t TimeMs)
{
    (void)p_platform;
    HAL_Delay(TimeMs);
    return 0;
}

void Reset_Sensor(VL53LMZ_Platform *p_platform)
{
    (void)p_platform;
    HAL_GPIO_WritePin(VL_XSHUT_GPIO_Port, VL_XSHUT_Pin, GPIO_PIN_RESET);
    HAL_Delay(10);
    HAL_GPIO_WritePin(VL_XSHUT_GPIO_Port, VL_XSHUT_Pin, GPIO_PIN_SET);
    HAL_Delay(25);
}
