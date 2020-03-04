import os
import argparse
import numpy as np
import sympy as sp

def puthw(file_obj, scope, num):
    file_obj.write("#define %sH %d\n" % (scope, num[0]))
    file_obj.write("#define %sW %d\n" % (scope, num[1]))
def get_numPARA(width, mac_latency):
    divs=np.array(sp.divisors(int(width)))
    return min(divs[divs>(width/mac_latency)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='genarator')
    parser.add_argument('--size', '-s', nargs='+', type=int, default=[256,256])
    args = parser.parse_args()
    shape = np.array(args.size)
    f = open("fmap_size/height_width.h", mode='w')

    f.write("// input map size in each layer\n")
    f.write("#define C0_KSIZE 3\n")
    f.write("#define C0_PAD 1\n")
    f.write("#define C0_STRIDE 2\n")
    stride=[2,1,1,2,1,2,1,2,1,1,1,1,1,1,1,1]
    st = 1
    dwin_size = []
    for i in(stride):
        st*=i;
        dwin_size.append(shape//st)

    puthw(f, "C0_IN", shape)
    puthw(f, "C0_OUT", shape//stride[0])

    for i in range(1,14):
        puthw(f, "DW%d_IN"  % i, dwin_size[i-1])
        puthw(f, "DW%d_OUT" % i, dwin_size[i])
        bnPARA = get_numPARA(dwin_size[i][1], 11)
    for i in range(1,15):
        puthw(f, "PW%d_IN"  % i, dwin_size[i])
        puthw(f, "PW%d_OUT" % i, dwin_size[i])
    fmap_size = dwin_size[-1]
    puthw(f, "OUT_", fmap_size)
    f.close()

