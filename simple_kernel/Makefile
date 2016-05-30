all: host emu

host: host.cpp
	g++ -g -o pipe.exe host.cpp -DFPGA_DEVICE -lOpenCL
emu: rm_xil device.cl
	xocc -g --xdevice xilinx:adm-pcie-7v3:1ddr:1.0 -o pipe.xclbin -t sw_emu device.cl
hw: rm_xil device.cl
	xocc -g --xdevice xilinx:adm-pcie-7v3:1ddr:1.0 -o pipe.xclbin -t hw device.cl
rm_xil:
	rm -rf .Xil
clean: rm_xil
	rm -f xocc*
	rm -f sdaccel_profile_summary.*
	rm -f shared*
