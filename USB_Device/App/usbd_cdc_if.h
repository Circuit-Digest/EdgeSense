/**
  ******************************************************************************
  * @file           : usbd_cdc_if.h
  * @brief          : Header for usbd_cdc_if.c
  ******************************************************************************
  */
#ifndef __USBD_CDC_IF_H__
#define __USBD_CDC_IF_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "usbd_cdc.h"

#define APP_RX_DATA_SIZE  1024
#define APP_TX_DATA_SIZE  1024

extern USBD_CDC_ItfTypeDef USBD_Interface_fops_FS;
extern volatile uint8_t g_usb_rx_char;
extern volatile uint8_t g_usb_rx_flag;

uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len);
void USB_CDC_TxPump(void);

#ifdef __cplusplus
}
#endif

#endif /* __USBD_CDC_IF_H__ */
