import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn

def attention(img, att_map, out1, out2, size):
    """
    Inputs:
    img -- original image
    att_map -- attention map, in this case shape (7,7) or (7,7,1,1) else.
    
    To visualize just imshow new_img
    """
    
    att_map = np.reshape(att_map, size)
    att_map = att_map.repeat(32, axis=0).repeat(32, axis=1)
    att_map = np.tile(np.expand_dims(att_map, 2),[1,1,3])
    att_map[:,:,1:] = 0
    # apply gaussian
    att_map = gaussian_filter(att_map, sigma=7)
    att_map = (att_map-att_map.min()) / att_map.max()
    att_map = np.uint8(255*att_map)
    att_map = cv2.resize(att_map, (img.shape[1], img.shape[0]))
    att_map = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)
    new_img = att_map*0.7 + img*0.3
    # new_img = att_map * 255
    # new_img = new_img.astype(np.uint8)
    cv2.imwrite(out1, new_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out2, img)
    
    return new_img

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)


    mods = list(model.modules())
    mods = list(mods[1].modules())
    out_sizes = []

    for i in range(3, len(mods)):
        print(i)
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        if type(out) == list:
            out = out[0]
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))