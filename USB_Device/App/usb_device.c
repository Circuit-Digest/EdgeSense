/**
  ******************************************************************************
  * @file           : usb_device.c
  * @brief          : USB Device initialization and startup
  ******************************************************************************
  */
#include "main.h"
#include "usb_device.h"
#include "usbd_core.h"
#include "usbd_desc.h"
#include "usbd_cdc.h"
#include "usbd_cdc_if.h"

USBD_HandleTypeDef hUsbDeviceFS;
extern PCD_HandleTypeDef hpcd_USB_DRD_FS;

void MX_USB_Device_Init(void)
{
  /* Link PCD handle with USB device handle */
  hpcd_USB_DRD_FS.pData = &hUsbDeviceFS;

  /* Init Device Library */
  if (USBD_Init(&hUsbDeviceFS, &FS_Desc, 0) != USBD_OK)
  {
    Error_Handler();
  }

  /* Register the CDC Class */
  if (USBD_RegisterClass(&hUsbDeviceFS, &USBD_CDC) != USBD_OK)
  {
    Error_Handler();
  }

  /* Register the Interface */
  if (USBD_CDC_RegisterInterface(&hUsbDeviceFS, &USBD_Interface_fops_FS) != USBD_OK)
  {
    Error_Handler();
  }

  /* Start the USB Device Core */
  if (USBD_Start(&hUsbDeviceFS) != USBD_OK)
  {
    Error_Handler();
  }
}
