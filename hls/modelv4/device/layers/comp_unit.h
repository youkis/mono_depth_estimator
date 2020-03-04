void dw_reshape(
		fmap_t tmp_x[SIMD_SIZE],
		const ap_uint<2> kx,
		const ap_uint<8> layer_id){
#pragma HLS INLINE

	if(kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=1; i--)
			tmp_x[i] = tmp_x[i-1];
		if(layer_id<5){
			tmp_x[0]=0;
		}else if(layer_id<11){
#pragma unroll
			for(int n=0;n<DOWN1;n++)
				tmp_x[n*(SIMD_SIZE/DOWN1)]=0;
		}else if(layer_id<15){
#pragma unroll
			for(int n=0;n<DOWN2;n++)
				tmp_x[n*(SIMD_SIZE/DOWN2)]=0;
		}else{
#pragma unroll
			for(int n=0;n<DOWN3;n++)
				tmp_x[n*(SIMD_SIZE/DOWN3)]=0;
		}
	}else if(kx==2){
#pragma unroll
		for (int i=1; i<SIMD_SIZE; i++)
			tmp_x[i-1] = tmp_x[i];
		if(layer_id<5){
			tmp_x[SIMD_SIZE-1]=0;
		}else if(layer_id<11){
#pragma unroll
			for(int n=0;n<DOWN1;n++)
				tmp_x[(n+1)*(SIMD_SIZE/DOWN1)-1]=0;
		}else if(layer_id<15){
#pragma unroll
			for(int n=0;n<DOWN2;n++)
				tmp_x[(n+1)*(SIMD_SIZE/DOWN2)-1]=0;
		}else{
#pragma unroll
			for(int n=0;n<DOWN3;n++)
				tmp_x[(n+1)*(SIMD_SIZE/DOWN3)-1]=0;
		}
	}
}

void down3x3_reshape(
		fmap_t tmp_x[SIMD_SIZE],
		const ap_uint<2> kx,
		const ap_uint<8> layer_id){
#pragma HLS INLINE

	if(kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=1; i--)
			tmp_x[i] = tmp_x[i-1];
		if(layer_id==0){
			tmp_x[0]=0;
		}else if(layer_id==5){
#pragma unroll
			for(int n=0;n<DOWN1;n++)
				tmp_x[n*(SIMD_SIZE/DOWN1)]=0;
		}else if(layer_id==11){
#pragma unroll
			for(int n=0;n<DOWN2;n++)
				tmp_x[n*(SIMD_SIZE/DOWN2)]=0;
		}else{
#pragma unroll
			for(int n=0;n<DOWN3;n++)
				tmp_x[n*(SIMD_SIZE/DOWN3)]=0;
		}
	}
}

void atrous_reshape(
		fmap_t tmpbuf[SIMD_SIZE],
		const ap_uint<2> kx,
		const ap_uint<5> dilate
		){
#pragma HLS INLINE

	if(dilate==DILATE0 && kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=DILATE0; i--) tmpbuf[i] = tmpbuf[i-DILATE0];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE0; i++) tmpbuf[n*(SIMD_SIZE/DOWN3)+i]=0;
	}else if(dilate==DILATE1 && kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=DILATE1; i--) tmpbuf[i] = tmpbuf[i-DILATE1];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE1; i++) tmpbuf[n*(SIMD_SIZE/DOWN3)+i]=0;
	}else if(dilate==DILATE2 && kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=DILATE2; i--) tmpbuf[i] = tmpbuf[i-DILATE2];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE2; i++) tmpbuf[n*(SIMD_SIZE/DOWN3)+i]=0;
	}else if(dilate==DILATE3 && kx==0){
#pragma unroll
		for (int i=SIMD_SIZE-1; i>=DILATE3; i--) tmpbuf[i] = tmpbuf[i-DILATE3];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE3; i++) tmpbuf[n*(SIMD_SIZE/DOWN3)+i]=0;
	}else if(dilate==2 && kx==2){
#pragma unroll
		for (int i=2; i<SIMD_SIZE; i++) tmpbuf[i-2] = tmpbuf[i];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE0; i++) tmpbuf[(n+1)*(SIMD_SIZE/DOWN3)+i-DILATE0]=0;
	}else if(dilate==DILATE0 && kx==2){
#pragma unroll
		for (int i=DILATE1; i<SIMD_SIZE; i++) tmpbuf[i-DILATE1] = tmpbuf[i];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE1; i++) tmpbuf[(n+1)*(SIMD_SIZE/DOWN3)+i-DILATE1]=0;
	}else if(dilate==DILATE2 && kx==2){
#pragma unroll
		for (int i=DILATE2; i<SIMD_SIZE; i++) tmpbuf[i-DILATE2] = tmpbuf[i];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE2; i++) tmpbuf[(n+1)*(SIMD_SIZE/DOWN3)+i-DILATE2]=0;
	}else if(dilate==DILATE3 && kx==2){
#pragma unroll
		for (int i=DILATE3; i<SIMD_SIZE; i++) tmpbuf[i-DILATE3] = tmpbuf[i];
#pragma unroll
		for(int n=0; n<DOWN3; n++)
#pragma unroll
			for(int i=0; i<DILATE3; i++) tmpbuf[(n+1)*(SIMD_SIZE/DOWN3)+i-DILATE3]=0;
	}

}

void pw_reshape(
		const fmap_t dup0[BUF_CH][SIMD_SIZE],
		fmap_t tmp_x[SIMD_SIZE],
		const ap_uint<16> ifeat_addr,
		const ap_uint<12> idx, const ap_uint<8> ofeat,
		const ap_uint<12> convloop
		){
#pragma HLS INLINE

	ap_uint<10> sp_if[DOWN3];
	ap_uint<16> sp_ify[DOWN3];
	ap_uint<8> sp_ifx[DOWN3];

	const ap_uint<16> y_idx=ofeat*convloop+idx;
	sp_if[0]= pw_ifs0[ifeat_addr+y_idx];
	sp_if[1]= pw_ifs1[ifeat_addr+y_idx];
	sp_if[2]= pw_ifs2[ifeat_addr+y_idx];
	sp_if[3]= pw_ifs3[ifeat_addr+y_idx];
	sp_if[4]= pw_ifs4[ifeat_addr+y_idx];
	sp_if[5]= pw_ifs5[ifeat_addr+y_idx];
	sp_if[6]= pw_ifs6[ifeat_addr+y_idx];
	sp_if[7]= pw_ifs7[ifeat_addr+y_idx];

#pragma unroll
	for (int n=0; n<DOWN3; n++){
		sp_ify[n] = (sp_if[n]>>DOWN3S);
		sp_ifx[n] = (sp_if[n]&(DOWN3-1));
	}

#pragma unroll
	for (int n=0; n<DOWN3; n++){
#pragma unroll
		for (int i=0; i<SIMD_SIZE/DOWN3; i++){
				tmp_x[n*(SIMD_SIZE/DOWN3)+i] = dup0[sp_ify[n]][sp_ifx[n]*(SIMD_SIZE/DOWN3)+i];
		}
	}
}


void init_linebuf3x3dwn(
		fmap_t in_buf[4*BUF_CH][SIMD_SIZE],
		const ap_uint<8> in_axis
		){
	for(int ky=0; ky<2; ky++){
#pragma unroll 1
		for (int ch=0; ch<in_axis/2; ch++){
			fmap_t tmp[2][SIMD_SIZE];
#pragma ii 1
			for (int mix=0; mix<2; mix++){
				fmap_t tmp2[SIMD_SIZE];
#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++)
					tmp2[i] = in_buf[BUF_CH*(ky+1)+ch*2+mix][i];
#pragma unroll
				for (int i=0; i<SIMD_SIZE/2; i++){
					tmp[0][mix*(SIMD_SIZE/2)+i] = tmp2[i*2];
					tmp[1][mix*(SIMD_SIZE/2)+i] = tmp2[i*2+1];
				}
			}
#pragma ii 1
			for (int mix=0; mix<2; mix++){
#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++)
					in_buf[BUF_CH*(ky+1)+ch*2+mix][i] = tmp[mix][i];
			}
		}
	}
}

void init_linebuf3x3(
		fmap_t main_buf[FMAP_H][SIMD_SIZE],
		fmap_t in_buf[4*BUF_CH][SIMD_SIZE],
		fmap_t dup0[BUF_CH][SIMD_SIZE],
		const ap_uint<8> y,
		const ap_uint<10> outh,
		const ap_uint<8> in_axis,
		const ap_uint<2> st,
		const ap_uint<3> lt,
		const ap_uint<8> dilate,
		const ap_uint<8> lid
		){
	const ap_uint<10> inh=inhs[lid];
	const ap_uint<16> src_addr=src_addrs[lid];
	ap_uint<10> addr;
	ap_uint<8> inaxis_addr;
	if(lid>=PPM_LAYER && lid<PPM_LAYER+8 && dilate==0)
		inaxis_addr=CBR0_DW_INCH/DOWN3;
	else
		inaxis_addr=in_axis;
	if(lt==0 || lt==1){
		addr = y * 2;
	} else if(lt==2 && y!=0){
		addr = y + 1;
	} else {
		addr = y;
	}
	//#pragma ii 1
	for (ap_int<4> ky=0; ky<3; ky++){
		for (int ch=0; ch<in_axis; ch++){
			fmap_t tmp[SIMD_SIZE];
			const ap_int<16> y_ctl=(ky-1)*dilate+addr+st;
			if(lt!=4 && ky!=2){
#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++)
					tmp[i] = in_buf[BUF_CH*(ky+1)+ch][i];
			}else if(y_ctl<0||y_ctl>=inh){
#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++)
					tmp[i]=0;
			}else{
#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++)
					tmp[i]=main_buf[src_addr+y_ctl*inaxis_addr+ch][i];
			}
#pragma unroll
			for (int i=0; i<SIMD_SIZE; i++){
				in_buf[BUF_CH*ky+ch][i]=tmp[i];
				dup0[ch][i]=tmp[i];
			}
		}
	}
}

void Conv(
		const ap_uint<8> lid, const ap_uint<3> lt, const ap_uint<5> dilate,
		fmap_t in_buf[4*BUF_CH][SIMD_SIZE],
		fmap_t dup0[BUF_CH][SIMD_SIZE],
		global const volatile transfer_w_t * restrict global_w,
		fmap_t wback[SIMD_SIZE],
		const ap_uint<8> ofeat){
	const ap_uint<12> convloop=convloops[lid];
	const ap_uint<12> bn_addr=bn_addrs[lid];
	const ap_uint<16> w_addr=w_addrs[lid];
	ap_uint<2> kx=0,ky=0;
	ap_uint<16> ifeat=0;
	mac_type reg[SIMD_SIZE];
	mac_type reg2[SIMD_SIZE];
	fmap_t act_val[SIMD_SIZE];

#pragma unroll
	for (int i=0; i<SIMD_SIZE; i++){
		reg[i]=0;
	}
#pragma ii 1
	for (ap_uint<12> idx=0; idx<convloop; idx++) {
		fmap_t tmp_x[SIMD_SIZE];
		fmap_t mac_x[SIMD_SIZE];
		ap_uint<8> y_addr;
		weight_type w[DOWN3];

		if(lt==0)             y_addr=(ifeat<<1)+(1^(kx&1));
		else if(lt==1)        y_addr=(ofeat<<1)+(1^(kx&1));
		else if(lt==2||lt==4) y_addr=ofeat;
		else                  y_addr=0;

#pragma unroll
		for (int i=0; i<SIMD_SIZE; i++) tmp_x[i] = in_buf[BUF_CH*ky+y_addr][i];

		if(lt==0||lt==1){
			down3x3_reshape(tmp_x, kx, lid);
		}else if(lt==2){
			dw_reshape(tmp_x, kx, lid);
		}else if(lt==3){
			pw_reshape(dup0, tmp_x, n_ifs[lid], idx, ofeat, convloop);
		}else if(lt==4){
			atrous_reshape(tmp_x, kx, dilate);
		}

#pragma unroll
    for (int i=0; i<SIMD_SIZE; i++){
      mac_x[i] = tmp_x[i];
    }

		const transfer_w_t tmp_w=global_w[w_addr+ofeat*convloop+idx];
		w[0]= (  tmp_w  ) & 0xFF;
		w[1]= (tmp_w>> 8) & 0xFF;
		w[2]= (tmp_w>>16) & 0xFF;
		w[3]= (tmp_w>>24) & 0xFF;
		w[4]= (tmp_w>>32) & 0xFF;
		w[5]= (tmp_w>>40) & 0xFF;
		w[6]= (tmp_w>>48) & 0xFF;
		w[7]= (tmp_w>>56) & 0xFF;

#pragma unroll
		for(int n=0;n<DOWN3;n++){
#pragma unroll
			for (int x=0; x<SIMD_SIZE/DOWN3; x++)
				reg[n*(SIMD_SIZE/DOWN3)+x] += (ap_int<16>)(mac_x[n*(SIMD_SIZE/DOWN3)+x]*w[n]);
		}

		kx++;
		if( kx==3 ){ kx=0; ky++; }
		if( ky==3 ){ ky=0; ifeat++; }
	}

#pragma unroll
	for (int i=0; i<SIMD_SIZE; i++){
		reg2[i] = reg[i];
	}
	// applying batch normalization
#pragma ii 1
	for (int n=0; n<DOWN3; n++) {
		const bn_scale_type scale=bn_scale[bn_addr+ofeat][n];
		const bn_shift_type shift=bn_shift[bn_addr+ofeat][n];
#pragma unroll
		for (int x=0; x<(SIMD_SIZE/DOWN3); x++) {
			short norm_x = ((short)(((long)reg2[n*(SIMD_SIZE/DOWN3)+x]*(long)scale)>>(BN_SCALE_EXP-BN_SHIFT_EXP))+shift)>>BN_SHIFT_EXP;
			// relu
			if(norm_x>255)    act_val[n*(SIMD_SIZE/DOWN3)+x] = 255;
			else if(norm_x>0) act_val[n*(SIMD_SIZE/DOWN3)+x] = norm_x;
			else              act_val[n*(SIMD_SIZE/DOWN3)+x] = 0;
		}
	}
#pragma unroll
	for (int i=0; i<SIMD_SIZE; i++) {
		wback[i] = act_val[i];
	}

}


void Convolution(
		const ap_uint<8> lid,
		fmap_t main_buf[FMAP_H][SIMD_SIZE],
		global const volatile transfer_w_t * restrict global_w
		){
	fmap_t in_buf[4*BUF_CH][SIMD_SIZE];
	fmap_t dup0[BUF_CH][SIMD_SIZE];
	const ap_uint<3> layer_type=ltypes[lid];
	const ap_uint<10> outh=ouths[lid];
	const ap_uint<8> axis_in=norm_inchs_addr[lid];
	const ap_uint<8> axis_out=norm_outchs[lid];
	const ap_uint<8> axis_out_addr=norm_outchs_addr[lid];
	const ap_uint<16> dst_addr=dst_addrs[lid];
	const ap_uint<5> dilate=dilates[lid];
#if 0
	printf("axis_in:%d, ",(int)axis_in);
	printf("axis_out:%d, axis_out_addr: %d\n",(int)axis_out, (int)axis_out_addr);
  printf("src: %d, dst: %d\n",(int)src_addrs[lid], (int)dst_addr);
#endif
#if 0
	if(lid==PPM_LAYER+8){
		unsigned cnt=0;
		for (int dst_y=0; dst_y<inhs[lid]; dst_y++)
			for (ap_uint<8> ofeat=0; ofeat<axis_in; ofeat++)
				for(int i=0; i<SIMD_SIZE; i++){
					//printf("%f\n",(float)main_buf[dst_addr+dst_y*axis_out_addr+ofeat][i]);
					//if(ofeat*DOWN2+(i/32)>=128/4*5)
					main_buf[src_addrs[lid]+dst_y*axis_in+ofeat][i]=ppmout[cnt];
					cnt++;
				}
	}
#endif
	// Initialization
	for (int ch=0; ch<4*BUF_CH; ch++){
#pragma unroll
		for (int x=0; x<SIMD_SIZE; x++) {
			in_buf[ch][x]=0;
		}
	}

	for (int dst_y=0; dst_y<outh; dst_y++) {
		ap_uint<2> st;
		if(layer_type==0 || layer_type==1 || (layer_type==2 && dst_y==0)) st=2;
		else st=1;

		for(int s=0;s<st;s++){
			init_linebuf3x3(main_buf, in_buf, dup0, dst_y, outh, axis_in, s, layer_type, dilate, lid);
		}
		if(layer_type==1){
			init_linebuf3x3dwn(in_buf, axis_in);
		}

		for (ap_uint<8> ofeat=0; ofeat<axis_out; ofeat++) {
#pragma HLS DATAFLOW
			Conv(lid, layer_type, dilate, in_buf, dup0, global_w, main_buf[dst_addr+dst_y*axis_out_addr+ofeat], ofeat);
		}
	}

#if 0
	if(lid==38){
		unsigned cnt=0;
		for (int dst_y=0; dst_y<outh; dst_y++)
			for (ap_uint<8> ofeat=0; ofeat<axis_out; ofeat++)
				for(int i=0; i<SIMD_SIZE; i++){
					printf("%d\n",(int)main_buf[dst_addr+dst_y*axis_out_addr+ofeat][i]);
				}
	}
#endif

#ifdef __DEBUG__
	printf("Convolutional layer%d end\n",(int)lid);
#endif
}


