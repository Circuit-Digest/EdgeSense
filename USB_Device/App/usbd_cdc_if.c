/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @brief          : Generic media access layer for USB CDC VCP
  ******************************************************************************
  */
#include "usbd_cdc_if.h"

USBD_CDC_LineCodingTypeDef LineCoding =
{
  115200, /* baud rate*/
  0x00,   /* stop bits-1*/
  0x00,   /* parity - none*/
  0x08    /* num of bits 8 */
};

uint8_t UserRxBufferFS[APP_RX_DATA_SIZE];
uint8_t UserTxBufferFS[APP_TX_DATA_SIZE];
volatile uint8_t g_usb_rx_char = 0;
volatile uint8_t g_usb_rx_flag = 0;

extern USBD_HandleTypeDef hUsbDeviceFS;

static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);

USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS
};

static int8_t CDC_Init_FS(void)
{
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  return (USBD_OK);
}

static int8_t CDC_DeInit_FS(void)
{
  return (USBD_OK);
}

static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  switch(cmd)
  {
    case CDC_SEND_ENCAPSULATED_COMMAND:
      break;

    case CDC_GET_ENCAPSULATED_RESPONSE:
      break;

    case CDC_SET_COMM_FEATURE:
      break;

    case CDC_GET_COMM_FEATURE:
      break;

    case CDC_CLEAR_COMM_FEATURE:
      break;

    case CDC_SET_LINE_CODING:
      LineCoding.bitrate = (uint32_t)(pbuf[0] | (pbuf[1] << 8) | (pbuf[2] << 16) | (pbuf[3] << 24));
      LineCoding.format = pbuf[4];
      LineCoding.paritytype = pbuf[5];
      LineCoding.datatype = pbuf[6];
      break;

    case CDC_GET_LINE_CODING:
      pbuf[0] = (uint8_t)(LineCoding.bitrate);
      pbuf[1] = (uint8_t)(LineCoding.bitrate >> 8);
      pbuf[2] = (uint8_t)(LineCoding.bitrate >> 16);
      pbuf[3] = (uint8_t)(LineCoding.bitrate >> 24);
      pbuf[4] = LineCoding.format;
      pbuf[5] = LineCoding.paritytype;
      pbuf[6] = LineCoding.datatype;
      break;

    case CDC_SET_CONTROL_LINE_STATE:
      break;

    case CDC_SEND_BREAK:
      break;

    default:
      break;
  }

  return (USBD_OK);
}

extern void AT_Cmd_FeedChar(char c);

static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  if (Len != NULL && *Len > 0) {
    for (uint32_t i = 0; i < *Len; i++) {
      AT_Cmd_FeedChar((char)Buf[i]);
    }
  }
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
}

#define USB_TX_RING_SIZE 16384
static uint8_t s_usb_tx_ring[USB_TX_RING_SIZE];
static volatile uint16_t s_usb_tx_head = 0;
static volatile uint16_t s_usb_tx_tail = 0;
static uint8_t s_usb_ep_buf[64];

void USB_CDC_TxPump(void)
{
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  if (hcdc == NULL || hcdc->TxState != 0) {
    return;
  }

  uint16_t head = s_usb_tx_head;
  uint16_t tail = s_usb_tx_tail;

  if (head == tail) {
    return;
  }

  uint16_t len = 0;
  while (tail != head && len < 64) {
    s_usb_ep_buf[len++] = s_usb_tx_ring[tail];
    tail = (tail + 1) % USB_TX_RING_SIZE;
  }
  s_usb_tx_tail = tail;

  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, s_usb_ep_buf, len);
  USBD_CDC_TransmitPacket(&hUsbDeviceFS);
}

uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len)
{
  if (Buf == NULL || Len == 0) {
    return USBD_OK;
  }

  for (uint16_t i = 0; i < Len; i++) {
    uint16_t next_head = (s_usb_tx_head + 1) % USB_TX_RING_SIZE;
    if (next_head == s_usb_tx_tail) {
      USB_CDC_TxPump();
      if (next_head == s_usb_tx_tail) {
        break; // buffer full
      }
    }
    s_usb_tx_ring[s_usb_tx_head] = Buf[i];
    s_usb_tx_head = next_head;
  }

  USB_CDC_TxPump();
  return USBD_OK;
}
