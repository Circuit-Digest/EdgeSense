/**
  ******************************************************************************
  * @file           : usb_device.h
  * @brief          : Header for usb_device.c
  ******************************************************************************
  */
#ifndef __USB_DEVICE_H__
#define __USB_DEVICE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h5xx.h"
#include "stm32h5xx_hal.h"
#include "usbd_def.h"

extern USBD_HandleTypeDef hUsbDeviceFS;

void MX_USB_Device_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* __USB_DEVICE_H__ */
