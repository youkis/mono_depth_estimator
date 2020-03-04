#!/bin/bash
sync () {
	if [ $1 = "fpga" ];then
		aoc -I $INTELFPGAOCLSDKROOT/include/kernel_headers -report -v -g ./device/mobilenetPSP.cl -o bin/mobilenetPSP.aocx -board=c5p
	fi
	if [ $1 = "cpu" ];then
		aoc -I $INTELFPGAOCLSDKROOT/include/kernel_headers -report -v -g ./device/mobilenetPSP.cl -o bin/mobilenetPSP.aocx -D EMU -D __DEBUG__ -march=emulator -emulator-channel-depth-model=strict
	fi
}
run () {
	if [ $1 = "fpga" ];then
		./bin/host
	fi
	if [ $1 = "cpu" ];then
		env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host
	fi
}

$1 $2
