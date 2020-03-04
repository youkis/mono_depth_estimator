import numpy as np
import os
from shutil import rmtree
SCALE_BIT=30
SHIFT_BIT=15
DOWN0=1
DOWN1=2
DOWN2=4
DOWN3=8
ifs_bits=9
w_type=(np.uint64, 'unsigned long')

def save_header(file_name,string_data,option='a'):
    print('\tINFO: Convert -> %s' % file_name)
    f = open(file_name, option)
    f.write(string_data)
    f.close()

def arr2header(arr, dtype='float', form=None, width=140):
    if form is not None: formatter={dtype:lambda x: form % x}
    elif dtype=='float': formatter={'float':lambda x: "%16.9e" % x}
    elif dtype=='uint64': formatter={'int':lambda x: hex(x)}
    else:                formatter={'int':lambda x: "%d" % x}
    return np.array2string(arr, threshold=int(2**31), separator=',', formatter=formatter, max_line_width=width).replace(']', '}').replace('[', '{')+';\n'

npz_bn=np.load('weights/batchnorm.npz')
npz_idx=np.load('weights/sp_index.npz')
npz_w=np.load('weights/w_params.npz')

save_path='converted'
if os.path.exists(save_path):
    rmtree(save_path)
os.mkdir(save_path)

def convert_weight(npz_w, scope, para, save_path, w):
    arr=npz_w[scope]
    arr=arr.reshape((arr.shape[0]//para,para,-1)).transpose((0,2,1)).reshape((-1,para))
    for j in range(para):
        shape=(scope, j, arr.shape[0])
        save_header(os.path.join(save_path,'w_params.h'), ('constant ap_int<8> %s_%d[%d]=\n' % shape)+arr2header(arr[:,j], 'int'))
    print('w_size:',w.shape[0])
    return np.append(w,np.repeat(arr,DOWN3//para,axis=1),axis=0)
def convert_spindex(npz_w, scope, para, save_path, pw_ifs):
    idx=npz_idx[scope]
    idx=idx.reshape((idx.shape[0]//para,para,-1)).transpose((0,2,1)).reshape((-1,para))
    max_val=0
    li=[]
    for j in range(para):
        shape=(int(np.ceil(np.log2(idx.max()+1))),scope,j,idx.shape[0])
        header=arr2header(idx[:,j], 'int')
        save_header(os.path.join(save_path,'sp_index.h'), ('constant ap_uint<%d> %s_%d[%d]=\n' % shape)+header)
        for k in range(DOWN3//para):
            #li.append(idx[:,j].copy())
            li.append(idx[:,j].copy()*(DOWN3//para)+k)
    print('ifs_size:',pw_ifs.shape[0])
    return np.append(pw_ifs,np.array(li).transpose((1,0)),axis=0)
    #return np.append(pw_ifs,np.repeat(idx,DOWN3//para,axis=1)*(DOWN3//para)+np.array([[i for i in range(DOWN3//para) for j in range(para)]]),axis=0)
def convert_bn(npz_bn, scope, para, precision, save_path, bn):
    idx=npz_bn[scope]
    idx=idx.reshape((-1,para))
    save_header(os.path.join(save_path,'batchnorm.h'), ('constant ap_int<%d> %s[%d][%d]=\n' % ((precision,scope)+idx.shape))+arr2header(idx, 'int'))
    if precision==SCALE_BIT:
        print('scale_size:',bn.shape[0])
    return np.append(bn,np.repeat(idx,DOWN3//para,axis=1),axis=0)

pw_ifs=np.empty((0,DOWN3))
w=np.empty((0,DOWN3))
scale=np.empty((0,DOWN3))
shift=np.empty((0,DOWN3))

# conv0 weight
w=convert_weight(npz_w,'w_conv0', DOWN0, save_path, w)
# conv0 bn
scale=convert_bn(npz_bn, 'scale_conv0', DOWN0, SCALE_BIT, save_path, scale)
shift=convert_bn(npz_bn, 'shift_conv0', DOWN0, SHIFT_BIT, save_path, shift)

for i in range(1, 14):
    if i<3:   para=DOWN0
    elif i<5: para=DOWN1
    elif i<7: para=DOWN2
    else:     para=DOWN3
    # dw
    w=convert_weight(npz_w,'w_dw'+str(i), para, save_path,w)
    scale=convert_bn(npz_bn, 'scale_dw'+str(i), para, SCALE_BIT, save_path, scale)
    shift=convert_bn(npz_bn, 'shift_dw'+str(i), para, SHIFT_BIT, save_path, shift)

    # pw
    w=convert_weight(npz_w,'sp_w_pw'+str(i), para, save_path, w)
    pw_ifs = convert_spindex(npz_w,'sp_if_pw'+str(i), para, save_path, pw_ifs)
    scale=convert_bn(npz_bn, 'scale_pw'+str(i), para, SCALE_BIT, save_path,scale)
    shift=convert_bn(npz_bn, 'shift_pw'+str(i), para, SHIFT_BIT, save_path,shift)
    if i==4:
        # dw
        w=convert_weight(npz_w,'w_skip_dw', para, save_path,w)
        scale=convert_bn(npz_bn, 'scale_skip_dw', para, SCALE_BIT, save_path, scale)
        shift=convert_bn(npz_bn, 'shift_skip_dw', para, SHIFT_BIT, save_path, shift)
        # pw
        w=convert_weight(npz_w,'sp_w_skip_pw', para, save_path, w)
        pw_ifs = convert_spindex(npz_w,'sp_if_skip_pw', para, save_path, pw_ifs)
        scale=convert_bn(npz_bn, 'scale_skip_pw', para, SCALE_BIT, save_path,scale)
        shift=convert_bn(npz_bn, 'shift_skip_pw', para, SHIFT_BIT, save_path,shift)


para=DOWN3
# aspp
for i in range(4):
    # pw
    w=convert_weight(npz_w,'sp_w_aspp_pw'+str(i), para, save_path, w)
    pw_ifs = convert_spindex(npz_w,'sp_if_aspp_pw'+str(i), para, save_path, pw_ifs)
    scale=convert_bn(npz_bn, 'scale_aspp_pw'+str(i), para, SCALE_BIT, save_path,scale)
    shift=convert_bn(npz_bn, 'shift_aspp_pw'+str(i), para, SHIFT_BIT, save_path,shift)
    # dw
    w=convert_weight(npz_w,'w_aspp_dw'+str(i), para, save_path,w)
    scale=convert_bn(npz_bn, 'scale_aspp_dw'+str(i), para, SCALE_BIT, save_path, scale)
    shift=convert_bn(npz_bn, 'shift_aspp_dw'+str(i), para, SHIFT_BIT, save_path, shift)
#cbr conv
w=convert_weight(npz_w,'w_cbr0_dw', para, save_path,w);
w=convert_weight(npz_w,'sp_w_cbr0_pw', para, save_path,w);
pw_ifs = convert_spindex(npz_w,'sp_if_cbr0_pw', para, save_path, pw_ifs)
w=convert_weight(npz_w,'w_cbr1_dw', para, save_path,w);
w=convert_weight(npz_w,'sp_w_cbr1_pw', para, save_path,w);
pw_ifs = convert_spindex(npz_w,'sp_if_cbr1_pw', para, save_path, pw_ifs)
scale=convert_bn(npz_bn, 'scale_cbr0_dw', para, SCALE_BIT, save_path,scale)
shift=convert_bn(npz_bn, 'shift_cbr0_dw', para, SHIFT_BIT, save_path,shift)
scale=convert_bn(npz_bn, 'scale_cbr0_pw', para, SCALE_BIT, save_path,scale)
shift=convert_bn(npz_bn, 'shift_cbr0_pw', para, SHIFT_BIT, save_path,shift)
scale=convert_bn(npz_bn, 'scale_cbr1_dw', para, SCALE_BIT, save_path,scale)
shift=convert_bn(npz_bn, 'shift_cbr1_dw', para, SHIFT_BIT, save_path,shift)
scale=convert_bn(npz_bn, 'scale_cbr1_pw', para, SCALE_BIT, save_path,scale)
shift=convert_bn(npz_bn, 'shift_cbr1_pw', para, SHIFT_BIT, save_path,shift)
#w.append([w_out for w_out in npz_w['w_out_main'].reshape(-1, para).transpose((1,0))])
#pw_ifs=pw_ifs+[np.arange(0,32,para)+i for i in range(para)]

lshift = np.arange(0,8*para,8).astype(w_type[0])[np.newaxis,:]
w=(w.astype(np.uint8).astype(w_type[0])<<lshift).sum(axis=1)

for i,ifs in enumerate(pw_ifs.transpose((1,0)).astype(np.int)):
    save_header(os.path.join(save_path,'ifs.h'), ('constant ap_uint<%d> pw_ifs%d[%d]=\n' % (ifs_bits, i, *ifs.shape))+arr2header(ifs, 'int'))
save_header(os.path.join(save_path,'w.h'), ('const %s weights[%d]=\n' % (w_type[1], *w.shape))+arr2header(w, 'uint64'))
save_header(os.path.join(save_path,'scale.h'), ('constant ap_int<%d> bn_scale[%d][%d]=\n' % (SCALE_BIT, *scale.shape))+arr2header(scale.astype(np.int), 'int'))
save_header(os.path.join(save_path,'shift.h'), ('constant ap_int<%d> bn_shift[%d][%d]=\n' % (SHIFT_BIT, *shift.shape))+arr2header(shift.astype(np.int), 'int'))

