/*
 * at_cmd.c
 * Asynchronous Non-Blocking AT Command Processor for Host Communication
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
  * @file    at_cmd.c
  * @brief   Asynchronous Non-Blocking AT Command Processor
  *          Supports: AT+MODE (GESTURE/POSTURE/SURFACE/SMOKE), AT+STREAM,
  *                    AT+RES, AT+PAUSE, AT+RESUME, AT+FPS, AT+SYSINFO, AT+HELP
  ******************************************************************************
  */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#include "main.h"
#include "at_cmd.h"
#include "pipeline_manager.h"

// External state from main.c
extern uint8_t g_ranging_active;
extern float g_fps;
extern void Set_Sensor_Ranging(uint8_t enable);

static char s_rx_line_buf[AT_CMD_BUFFER_SIZE];
static uint8_t s_rx_idx = 0;
static char s_pending_cmd[AT_CMD_BUFFER_SIZE];
static volatile bool s_cmd_pending = false;

static void str_toupper(char *s) {
    while (*s) {
        *s = (char)toupper((unsigned char)*s);
        s++;
    }
}

void AT_Cmd_Init(void)
{
    s_rx_idx = 0;
    s_cmd_pending = false;
    memset(s_rx_line_buf, 0, sizeof(s_rx_line_buf));
    memset(s_pending_cmd, 0, sizeof(s_pending_cmd));
}

void AT_SendResponse(const char *fmt, ...)
{
    char out_buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(out_buf, sizeof(out_buf), fmt, args);
    va_end(args);

    printf("%s", out_buf);
}

static void execute_at_command(char *cmd_line)
{
    // Trim leading whitespace
    while (*cmd_line == ' ' || *cmd_line == '\t') cmd_line++;

    // Convert to uppercase for matching
    char cmd_upper[AT_CMD_BUFFER_SIZE];
    strncpy(cmd_upper, cmd_line, sizeof(cmd_upper) - 1);
    cmd_upper[sizeof(cmd_upper) - 1] = '\0';
    str_toupper(cmd_upper);

    if (strcmp(cmd_upper, "AT") == 0) {
        AT_SendResponse("OK\r\n");
        return;
    }

    if (strcmp(cmd_upper, "AT+HELP") == 0 || strcmp(cmd_upper, "AT+?") == 0) {
        AT_SendResponse("+HELP: EdgeSense AT Command Set\r\n");
        AT_SendResponse("  AT                  - Ping device (returns OK)\r\n");
        AT_SendResponse("  AT+MODE?            - Query active application mode\r\n");
        AT_SendResponse("  AT+MODE=<MODE>      - Switch mode: GESTURE, POSTURE, SURFACE, SMOKE\r\n");
        AT_SendResponse("  AT+RES?             - Query sensor grid resolution and CNH bins\r\n");
        AT_SendResponse("  AT+RES=<4X4|8X8>    - Switch ToF profile: 4X4 (48 bins @ 25Hz), 8X8 (16 bins @ 15Hz)\r\n");
        AT_SendResponse("  AT+STREAM?          - Query streaming format\r\n");
        AT_SendResponse("  AT+STREAM=<FORMAT>  - Set stream: RESULTS, RAW, DEBUG, OFF\r\n");
        AT_SendResponse("  AT+STATUS?          - Query runtime status, FPS, and AI latency\r\n");
        AT_SendResponse("  AT+PAUSE            - Pause ToF sensor ranging\r\n");
        AT_SendResponse("  AT+RESUME           - Resume ToF sensor ranging\r\n");
        AT_SendResponse("OK\r\n");
        return;
    }

    if (strcmp(cmd_upper, "AT+MODE?") == 0) {
        AT_SendResponse("+MODE: %s\r\nOK\r\n", Pipeline_GetModeName(Pipeline_GetMode()));
        return;
    }

    if (strncmp(cmd_upper, "AT+MODE=", 8) == 0) {
        char *arg = cmd_upper + 8;
        if (strcmp(arg, "GESTURE") == 0) {
            Pipeline_SetMode(SENSE_MODE_GESTURE);
            if (Get_Sensor_Resolution() != 64) {
                Set_Sensor_Profile(64, 16, 15, 4);
            }
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "POSTURE") == 0) {
            Pipeline_SetMode(SENSE_MODE_POSTURE);
            if (Get_Sensor_Resolution() != 64) {
                Set_Sensor_Profile(64, 16, 15, 4);
            }
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "SURFACE") == 0) {
            Pipeline_SetMode(SENSE_MODE_SURFACE);
            if (Get_Sensor_Resolution() != 16 || Get_Sensor_NumBins() != 48) {
                Set_Sensor_Profile(16, 48, 25, 1);
            }
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "SMOKE") == 0) {
            Pipeline_SetMode(SENSE_MODE_SMOKE);
            if (Get_Sensor_Resolution() != 16 || Get_Sensor_NumBins() != 48) {
                Set_Sensor_Profile(16, 48, 25, 1);
            }
            AT_SendResponse("OK\r\n");
        } else {
            AT_SendResponse("ERROR: INVALID_MODE (Use GESTURE, POSTURE, SURFACE, SMOKE)\r\n");
        }
        return;
    }

    if (strcmp(cmd_upper, "AT+RES?") == 0) {
        uint8_t res = Get_Sensor_Resolution();
        uint16_t bins = Get_Sensor_NumBins();
        uint8_t freq = Get_Sensor_Frequency();
        uint8_t sub = Get_Sensor_SubSample();
        AT_SendResponse("+RES: RESOLUTION=%s,ZONES=%u,BINS=%u,FREQ=%u,SUBSAMPLE=%u\r\nOK\r\n",
                       (res == 16) ? "4X4" : "8X8", res, bins, freq, sub);
        return;
    }

    if (strncmp(cmd_upper, "AT+RES=", 7) == 0) {
        char *arg = cmd_upper + 7;
        uint8_t status = 0;
        if (strcmp(arg, "4X4") == 0 || strcmp(arg, "16") == 0) {
            // 4x4 mode: 16 zones, 48 bins, 25 Hz, sub_sample 1
            status = Set_Sensor_Profile(16, 48, 25, 1);
            if (status == 0) {
                AT_SendResponse("OK\r\n");
            } else {
                AT_SendResponse("ERROR: RECONFIG_FAILED (0x%02X)\r\n", status);
            }
        } else if (strcmp(arg, "8X8") == 0 || strcmp(arg, "64") == 0) {
            // 8x8 mode: 64 zones, 16 bins, 15 Hz, sub_sample 4
            status = Set_Sensor_Profile(64, 16, 15, 4);
            if (status == 0) {
                AT_SendResponse("OK\r\n");
            } else {
                AT_SendResponse("ERROR: RECONFIG_FAILED (0x%02X)\r\n", status);
            }
        } else {
            AT_SendResponse("ERROR: INVALID_RES (Use 4X4 or 8X8)\r\n");
        }
        return;
    }

    if (strcmp(cmd_upper, "AT+STREAM?") == 0) {
        AT_SendResponse("+STREAM: %s\r\nOK\r\n", Pipeline_GetStreamModeName(Pipeline_GetStreamMode()));
        return;
    }

    if (strncmp(cmd_upper, "AT+STREAM=", 10) == 0) {
        char *arg = cmd_upper + 10;
        if (strcmp(arg, "RESULTS") == 0) {
            Pipeline_SetStreamMode(STREAM_MODE_RESULTS);
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "RAW") == 0) {
            Pipeline_SetStreamMode(STREAM_MODE_RAW);
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "DEBUG") == 0) {
            Pipeline_SetStreamMode(STREAM_MODE_DEBUG);
            AT_SendResponse("OK\r\n");
        } else if (strcmp(arg, "OFF") == 0) {
            Pipeline_SetStreamMode(STREAM_MODE_OFF);
            AT_SendResponse("OK\r\n");
        } else {
            AT_SendResponse("ERROR: INVALID_STREAM (Use RESULTS, RAW, DEBUG, OFF)\r\n");
        }
        return;
    }

    if (strcmp(cmd_upper, "AT+STATUS?") == 0) {
        AT_SendResponse("+STATUS: MODE=%s,STREAM=%s,RES=%s,FPS=%.1f,RANGING=%u,LATENCY_US=%lu\r\nOK\r\n",
                       Pipeline_GetModeName(Pipeline_GetMode()),
                       Pipeline_GetStreamModeName(Pipeline_GetStreamMode()),
                       (Get_Sensor_Resolution() == 16) ? "4X4" : "8X8",
                       g_fps,
                       g_ranging_active,
                       Pipeline_GetLatestLatencyUs());
        return;
    }

    if (strcmp(cmd_upper, "AT+PAUSE") == 0) {
        Set_Sensor_Ranging(0);
        AT_SendResponse("OK\r\n");
        return;
    }

    if (strcmp(cmd_upper, "AT+RESUME") == 0) {
        Set_Sensor_Ranging(1);
        AT_SendResponse("OK\r\n");
        return;
    }

    AT_SendResponse("ERROR: UNKNOWN_CMD (%s)\r\n", cmd_line);
}

void AT_Cmd_FeedChar(char c)
{
    if (c == '\r' || c == '\n') {
        if (s_rx_idx > 0) {
            s_rx_line_buf[s_rx_idx] = '\0';
            if (!s_cmd_pending) {
                strncpy(s_pending_cmd, s_rx_line_buf, sizeof(s_pending_cmd) - 1);
                s_pending_cmd[sizeof(s_pending_cmd) - 1] = '\0';
                s_cmd_pending = true;
            }
            s_rx_idx = 0;
        }
        return;
    }

    // Buffer printable character
    if (s_rx_idx < AT_CMD_BUFFER_SIZE - 1) {
        s_rx_line_buf[s_rx_idx++] = c;
    } else {
        // Overflow protection: reset buffer
        s_rx_idx = 0;
    }
}

void AT_Cmd_Process(void)
{
    if (!s_cmd_pending) return;

    char cmd[AT_CMD_BUFFER_SIZE];
    strncpy(cmd, s_pending_cmd, sizeof(cmd) - 1);
    cmd[sizeof(cmd) - 1] = '\0';
    s_cmd_pending = false;

    // Trim leading whitespace
    char *p = cmd;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '\0') {
        if ((p[0] == 'A' || p[0] == 'a') && (p[1] == 'T' || p[1] == 't')) {
            execute_at_command(p);
        } else {
            AT_SendResponse("ERROR: INVALID_CMD (Prefix command with AT, type AT+HELP)\r\n");
        }
    }
}
