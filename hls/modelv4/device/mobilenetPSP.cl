#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "mobilenetPSP.hpp"

typedef ap_uint<8> fmap_t;
typedef ap_int<8> weight_type;
typedef ap_int<32> mac_type;
typedef ap_int<30> bn_scale_type;
typedef ap_int<15> bn_shift_type;
#define BN_SCALE_EXP 23
#define BN_SHIFT_EXP 5


#define MAX_FMAP_SIZE (DW1_INCH*DW1_INW*DW1_INH)
#define SIMD_SIZE C0_OUTW
#define FMAP_H (MAX_FMAP_SIZE/SIMD_SIZE)
#define DOWN0S 0
#define DOWN1S 1
#define DOWN2S 2
#define DOWN3S 3
#define DOWN0 (1<<DOWN0S)
#define DOWN1 (1<<DOWN1S)
#define DOWN2 (1<<DOWN2S)
#define DOWN3 (1<<DOWN3S)
#define BUF_CH (512/DOWN3)

#define N_LAYERS 41
#define PPM_LAYER 29

#define DILATE0 2
#define DILATE1 3
#define DILATE2 6
#define DILATE3 12

/* fifo */
typedef channel fmap_t fifo_t;
typedef channel mac_type bn_fifo_t;
typedef channel weight_type w_fifo_t;

#include "weights/ifs.h"
#include "weights/scale.h"
#include "weights/shift.h"
#include "weights/n_loop.h"

#include "layers/layer_type.h"
#include "layers/comp_unit.h"

void load_fmap(
		global const volatile transfer_t * restrict in,
		fmap_t main_buffer[FMAP_H][SIMD_SIZE]){
	fmap_t pack[C0_OUTW];
	for(int i=0; i<C0_INCH*C0_INH*2; i++){
#pragma ii 1
		for(int j=0; j<C0_OUTW/8; j++){
			transfer_t tmp = in[i*(C0_OUTW/8)+j];
			pack[j*8  ]= (  tmp  ) & 0xFF; pack[j*8+1]= (tmp>> 8) & 0xFF;
			pack[j*8+2]= (tmp>>16) & 0xFF; pack[j*8+3]= (tmp>>24) & 0xFF;
			pack[j*8+4]= (tmp>>32) & 0xFF; pack[j*8+5]= (tmp>>40) & 0xFF;
			pack[j*8+6]= (tmp>>48) & 0xFF; pack[j*8+7]= (tmp>>56) & 0xFF;
		}

#pragma unroll
		for(int j=0; j<C0_OUTW; j++){
			main_buffer[src_addrs[0]+i][j] = pack[j];
		}
	}
#ifdef __DEBUG__
	printf("loaded an input image to local memory\n");
#endif
}

__kernel
void PSPNet(
		global const volatile transfer_t * restrict global_buf_in_c0,
		global transfer_t * restrict out_img,
		global const volatile transfer_w_t * restrict global_w
		){
	fmap_t fmap_buf[FMAP_H][SIMD_SIZE];
#ifdef __DEBUG__
	printf("== PSPNet ==\n");
#endif

	load_fmap(global_buf_in_c0, fmap_buf);

	/* MobileNetv1 1.0 */
	for(ap_uint<8> lid=0; lid<N_LAYERS; lid++){
		Convolution(lid, fmap_buf, global_w);
	}

	for(int i=0; i<N_OUTIMG/SIMD_SIZE; i++){
		ap_uint<16> addr;
		fmap_t tmp[SIMD_SIZE];
		if(i<(N_MAIN_FMAP/SIMD_SIZE))
			addr=0;
		else
			addr=SKIP_ADDR-(OUT_W*OUT_H*MAIN_OUTCH/SIMD_SIZE);
#pragma unroll
		for(int j=0; j<SIMD_SIZE; j++){
			tmp[j] = fmap_buf[addr+i][j];
		}
#pragma ii 1
		for(int j=0; j<SIMD_SIZE/8; j++){
			transfer_t pack=0;
			pack = (transfer_t)tmp[j*8+0] |
			       (transfer_t)tmp[j*8+1]<<8 |
			       (transfer_t)tmp[j*8+2]<<16 |
			       (transfer_t)tmp[j*8+3]<<24 |
			       (transfer_t)tmp[j*8+4]<<32 |
			       (transfer_t)tmp[j*8+5]<<40 |
			       (transfer_t)tmp[j*8+6]<<48 |
			       (transfer_t)tmp[j*8+7]<<56;
			out_img[i*(SIMD_SIZE/8)+j] = pack;
		}
	}
}

