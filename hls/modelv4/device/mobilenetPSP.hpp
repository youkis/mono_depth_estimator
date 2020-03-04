#ifndef MOBILENETPSP_H
#define MOBILENETPSP_H

// input image size
typedef unsigned char in_img_type;
typedef unsigned char out_t;
typedef unsigned long transfer_t;
typedef unsigned long transfer_w_t;


#include "fmap_size/height_width.h"
#include "fmap_size/channel.h"

#define N_MAIN_FMAP  (OUT_H*SKIP_OUTCH*OUT_W)
#define N_SKIP_FMAP  (PW3_OUTH*SKIP_OUTCH*PW3_OUTW)

#define N_OUTIMG  (N_MAIN_FMAP+N_SKIP_FMAP)
#define N_INIMG   (C0_INH*C0_INW*C0_INCH)

#define N_C0INBUF (N_INIMG/8)
#define N_OUTBUF  (N_OUTIMG/8)

#define N_W 35240

#define PUSH(que, val) write_channel_intel(que,val);
#define POP(val, que) val=read_channel_intel(que);

#endif
