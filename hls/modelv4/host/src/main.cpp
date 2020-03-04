#include<iostream>
#include<fstream>
#include<string>
#include<cstdlib>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<signal.h> // for signal handle

#include"mobilenetPSP.hpp"
#include"CL/opencl.h"
#include"AOCLUtils/aocl_utils.h"
#include"w.h"

using namespace aocl_utils;
using namespace std;

cl_platform_id   platform = NULL;
cl_device_id     device;
cl_context       context = NULL;
cl_command_queue queue_cnn;
cl_program       program = NULL;
cl_kernel        kernel_cnn;
cl_mem           w_buf;
cl_mem           inp_buf;
cl_mem           out_buf;
cl_event         kevent;

void resize_weighted_average(transfer_t *in, float *outmain, float *outval);
void save_result_txt(const char* fname, float *out_img);
bool init_opencl();
void cleanup();
volatile sig_atomic_t flag = 0; // for signal handle
void got_sig(int); // for signal handle

template<int CH, int INH, int INW, int OUTH, int OUTW>
void resize_ppm(out_t in[CH*INH*INW], float out[CH*OUTH*OUTW]){
  int idx,c,n,m;
  for(idx=0,c=0,n=0,m=0; idx<CH*OUTH*OUTW; idx++,m++){
    if(m==OUTW){ m=0; n++; }
    if(n==OUTH){ n=0; c++; }
    float om=(float)(m*(INW-1))/(OUTW-1);// -1 ajusted for chainer
    float on=(float)(n*(INH-1))/(OUTH-1);
    int m0=(int)om; int m1=m0+1;
    int n0=(int)on; int n1=n0+1;
    float dm=om-m0;
    float dn=on-n0;
    if(m1==INW) m1=INW-1;
    if(n1==INH) n1=INH-1;

    out[c*OUTH*OUTW+n*OUTW+m]
      = in[n1*CH*INW+c*INW+m1] *dm*dn
      + in[n0*CH*INW+c*INW+m1] *dm*(1-dn)
      + in[n1*CH*INW+c*INW+m0] *(1-dm)*dn
      + in[n0*CH*INW+c*INW+m0] *(1-dm)*(1-dn);
  }
}

//--------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------
int main( int argc, char *argv[]){

	transfer_t *global_c0in = (transfer_t *)alignedMalloc(N_C0INBUF*sizeof(transfer_t));
	transfer_w_t *global_w = (transfer_w_t *)alignedMalloc(N_W*sizeof(transfer_w_t));
	transfer_t *out_img = (transfer_t *)alignedMalloc(N_OUTBUF*sizeof(transfer_t));

	float *outmain = (float *)malloc(OUTMAIN_INCH*C0_INH*C0_INW*sizeof(float));
	float outval[C0_INH*C0_INW];
	cl_int status;
	cl_event write_event, kernel_event;
	if (!init_opencl()) return -1;

	// load image ----------------------------------------------------
	char fname[256];
	FILE *fp;
	sprintf( fname, "tmp_img.txt");
	fp = fopen( fname, "r");
	for(int ifeat = 0; ifeat < C0_INCH; ifeat++) {
		for(int y = 0; y < C0_INH; y++) {
			in_img_type in_buf[C0_INW];
			for(int x=0; x<C0_INW/2; x++){
				for(int st=0; st<2; st++){
					unsigned int imgval;
					int retval = fscanf( fp, "%u", &imgval);
					in_buf[st*(C0_INW/2)+x]=(in_img_type)imgval;
				}
			}
			memcpy(global_c0in + y*C0_INCH*2*(C0_OUTW/8) + ifeat*2*(C0_OUTW/8), (transfer_t *)in_buf, C0_INW);
		}
	}
	fclose(fp);

	// send weights
	for(int i=0;i<N_W;i++) global_w[i]=weights[i];
	status = clEnqueueWriteBuffer(queue_cnn, w_buf, CL_TRUE, 0, N_W*sizeof(transfer_w_t), (void *)global_w, 0, NULL, NULL);

	// Call CNN hardware
	status = clEnqueueWriteBuffer(queue_cnn, inp_buf, CL_FALSE, 0, N_C0INBUF*sizeof(transfer_t), (void *)global_c0in, 0, NULL, &write_event);
	checkError(status, "Failed to transfer input");
	status = clSetKernelArg(kernel_cnn,  0, sizeof(cl_mem), &inp_buf);
	status = clSetKernelArg(kernel_cnn,  1, sizeof(cl_mem), &out_buf);
	status = clSetKernelArg(kernel_cnn,  2, sizeof(cl_mem), &w_buf);
	status = clEnqueueTask(queue_cnn, kernel_cnn, 1, &write_event, &kernel_event);
	status = clWaitForEvents(1, &kernel_event); // for printing fps
	checkError(status, "Failed to finish event");

	cl_ulong start, end;
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	printf("%f [us]\n", (double)(end-start)*1e-3f);

	status = clEnqueueReadBuffer(queue_cnn, out_buf, CL_TRUE, 0, N_OUTBUF*sizeof(transfer_t), out_img, 1, &kernel_event, NULL);
	checkError(status, "Failed to copy data from device");
	clReleaseEvent(write_event);
	clReleaseEvent(kernel_event);

	resize_weighted_average(out_img, outmain, outval);
	save_result_txt("result.txt", outval);
	//cv::imwrite("pred.jpg", image);
	//cv::namedWindow("Image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	//cv::imshow("Image", pred_img);
	//cv::waitKey(1000);

	alignedFree(global_c0in);
	alignedFree(out_img);
	free(outmain);
	return 0;
}

// last convolution
void resize_weighted_average(transfer_t in[N_OUTBUF], float outmain[OUTMAIN_INCH*C0_INH*C0_INW], float outval[C0_INH*C0_INW]){
	const char w_out_main[128]= {127,90,70,57,55,107,117,127,98,115,118,65,109,127,-76,127,126,65,50,85,92,75,122,121,80,125,61,
	97,45,127,0,0,59,108,-34,127,127,121,70,27,76,-39,121,69,57,91,54,55,0,87,64,57,79,74,83,87,54,69,45,75,69,82,127,124,25,-22,
	-35,54,-22,-17,-20,10,-18,13,41,17,-27,19,17,21,17,16,-18,-25,40,-41,-14,20,-22,-41,-23,-16,-23,-20,-19,39,-31,13,11,27,-25,
	37,-25,23,-21,-32,-23,16,-43,-24,15,-13,28,-23,-24,37,-30,44,-20,30,-35,26,-37,11,-32,18,15,19};
	const float scale_out_main=5.449687365e-05;
	const float shift_out_main=8.359063864e-01;

	out_t *unpack = (out_t*)in;

	resize_ppm<MAIN_OUTCH, OUT_H, OUT_W, C0_INH, C0_INW> (unpack+0, outmain+0);
	resize_ppm<SKIP_OUTCH, PW3_OUTH, PW3_OUTW, C0_INH, C0_INW> (unpack+N_MAIN_FMAP, outmain+(MAIN_OUTCH*C0_INH*C0_INW));
	for(int y=0; y<C0_INH; y++){
		for(int x=0; x<C0_INW; x++){
			outval[y*C0_INW+x]=0;
		}
	}
	for(int ch=0; ch<OUTMAIN_INCH; ch++){
		for(int y=0; y<C0_INH; y++){
			for(int x=0; x<C0_INW; x++){
				outval[y*C0_INW+x] += w_out_main[ch]*outmain[ch*C0_INH*C0_INW+y*C0_INW+x];
			}
		}
	}
	for(int y=0; y<C0_INH; y++){
		for(int x=0; x<C0_INW; x++){
			outval[y*C0_INW+x]*=scale_out_main;
			outval[y*C0_INW+x]+=shift_out_main;
		}
	}
}
// save class prob into text file
void save_result_txt(const char* fname, float out_img[C0_INH*C0_INW]){
	FILE *fp;
	fp = fopen( fname, "w");
	for(int y=0; y<C0_INH; y++){
		for(int x=0; x<C0_INW; x++){
			fprintf( fp, "%f\n", out_img[y*C0_INW+x]);
		}
	}
	fclose(fp);
}

//------ bool init_opencl -------{{{
bool init_opencl() 
{
  cl_int status;
  unsigned int num_devices;
	printf("Initializing OpenCL\n");

	if(!setCwdToExeDir()) {
	  return false;
	}

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  cl_device_id *ids = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  if (num_devices != 1) {
    printf("ERROR: invalid device number %d\n", num_devices);
    return false;
  }
  device = ids[0];

  // Create the context.
  context = clCreateContext(NULL, num_devices, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("mobilenetPSP", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");


  // Command queue.
  queue_cnn = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  const char *kernel_name = "PSPNet";
  kernel_cnn = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel_cnn");

  // Input buffers.
  inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, N_C0INBUF*sizeof(transfer_t), NULL, &status);
  checkError(status, "Failed to create buffer for input");

  // Output buffer.
  out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N_OUTBUF*sizeof(transfer_t), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  // weights buffers.
  w_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, N_W*sizeof(transfer_w_t), NULL, &status);
  checkError(status, "Failed to create buffer for input");
  return true;
}
//}}}

void cleanup() {
  clReleaseKernel(kernel_cnn);
  clReleaseCommandQueue(queue_cnn);
  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
  clReleaseProgram(program);
  clReleaseContext(context);
}
void got_sig(int sig){ // can be called asynchronously
  flag = 1; // set flag
}
