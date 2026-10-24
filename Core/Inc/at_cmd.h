/*
 * at_cmd.h
 * Header for Asynchronous AT Command Processor
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
  * @file    at_cmd.h
  * @brief   Asynchronous Non-Blocking AT Command Processor Header
  ******************************************************************************
  */

#ifndef __AT_CMD_H
#define __AT_CMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define AT_CMD_BUFFER_SIZE 128

/**
  * @brief  Initializes the AT command processor
  */
void AT_Cmd_Init(void);

/**
  * @brief  Feed a received character into the AT command buffer
  * @param  c Incoming byte from UART or USB CDC
  */
void AT_Cmd_FeedChar(char c);

/**
  * @brief  Process pending AT commands in main thread context
  */
void AT_Cmd_Process(void);

/**
  * @brief  Send formatted response over USB CDC and UART
  * @param  fmt Printf-style format string
  */
void AT_SendResponse(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* __AT_CMD_H */
