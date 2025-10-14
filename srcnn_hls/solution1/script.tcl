############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project srcnn_hls
set_top srcnn
add_files src/conv1.cpp
add_files src/srcnn.cpp
add_files src/srcnn.h
add_files -tb test/set14 -cflags "-Wno-unknown-pragmas"
add_files -tb test/set5 -cflags "-Wno-unknown-pragmas"
add_files -tb src/weights -cflags "-Wno-unknown-pragmas"
add_files -tb test/util.cpp -cflags "-Isrc -Wno-unknown-pragmas"
add_files -tb test/util.h -cflags "-Isrc -Wno-unknown-pragmas"
add_files -tb test/tb_set14.cpp -cflags "-Isrc -Wno-unknown-pragmas"
add_files -tb test/tb_conv1.cpp -cflags "-Isrc -Wno-unknown-pragmas"
add_files -tb test/tb_srcnn.cpp -cflags "-Isrc -Wno-unknown-pragmas"
add_files -tb test/csim.cpp -cflags "-Isrc -Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
#source "./srcnn_hls/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
