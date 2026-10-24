/**
  ******************************************************************************
  * @file           : usbd_desc.h
  * @brief          : Header for usbd_desc.c
  ******************************************************************************
  */
#ifndef __USBD_DESC_H
#define __USBD_DESC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "usbd_def.h"

#define USBD_VID                      0x0483
#define USBD_PID_FS                   0x5740
#define USBD_LANGID_STRING            1033
#define USBD_MANUFACTURER_STRING      "STMicroelectronics"
#define USBD_PRODUCT_STRING_FS        "EdgeSense Virtual ComPort"
#define USBD_CONFIGURATION_STRING_FS  "CDC Config"
#define USBD_INTERFACE_STRING_FS      "CDC Interface"

extern USBD_DescriptorsTypeDef FS_Desc;

#ifdef __cplusplus
}
#endif

#endif /* __USBD_DESC_H */
