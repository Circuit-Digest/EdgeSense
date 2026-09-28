/**
  ******************************************************************************
  * @file           : usbd_conf.h
  * @brief          : Configuration file for USB Device Library on STM32H5
  ******************************************************************************
  */
#ifndef __USBD_CONF_H
#define __USBD_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h5xx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USBD_MAX_NUM_INTERFACES     1U
#define USBD_MAX_NUM_CONFIGURATION  1U
#define USBD_MAX_STR_DESC_SIZ       0x100U
#define USBD_SELF_POWERED           1U
#define USBD_DEBUG_LEVEL            0U

#define USBD_SUPPORT_USER_STRING_DESC 0U
#define USBD_CLASS_USER_STRING_DESC   0U
#define USBD_CLASS_BOS_ENABLED        0U
#define USB_BB_MAX_NUM_ALT_MODE       0x2U

#define USBD_CDC_INTERVAL             2000U

/* Memory management macros */
#define USBD_malloc                   (void *)USBD_static_malloc
#define USBD_free                     USBD_static_free
#define USBD_memset                   memset
#define USBD_memcpy                   memcpy
#define USBD_Delay                    HAL_Delay

/* DEBUG macros */
#if (USBD_DEBUG_LEVEL > 0U)
#define  USBD_UsrLog(...)   printf(__VA_ARGS__);\
                            printf("\n");
#else
#define USBD_UsrLog(...)
#endif

#if (USBD_DEBUG_LEVEL > 1U)
#define  USBD_ErrLog(...)   printf("ERROR: ") ;\
                            printf(__VA_ARGS__);\
                            printf("\n");
#else
#define USBD_ErrLog(...)
#endif

#if (USBD_DEBUG_LEVEL > 2U)
#define  USBD_DbgLog(...)   printf("DEBUG : ") ;\
                            printf(__VA_ARGS__);\
                            printf("\n");
#else
#define USBD_DbgLog(...)
#endif

void *USBD_static_malloc(uint32_t size);
void USBD_static_free(void *p);

#ifdef __cplusplus
}
#endif

#endif /* __USBD_CONF_H */
