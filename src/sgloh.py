import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from kornia_moons.feature import opencv_kpts_from_laf
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def laf2homo(kps):
    c = kps[:, :, 2]
    s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps / s.reshape(-1, 1, 1)
    Hi[:, 2, 2] = 1 

    H = torch.linalg.inv(Hi)
    
    return c, H, s ** 0.5


def get_inverse(pt1, Hs1):
            
    pt1_ = Hs1.bmm(torch.hstack((pt1, torch.ones((pt1.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt1_ = pt1_[:, :2] / pt1_[:, 2].unsqueeze(-1)
    
    Hi1 = torch.linalg.inv(Hs1).squeeze(1)    
    
    return pt1_, Hi1


def go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_diff_', stretch=False):        
    # warning image must be grayscale and not rgb!

    pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1, pt2, Hs) 
            
    patch1 = patchify(im1, pt1_, Hi1, w)
    patch2 = patchify(im2, pt2_, Hi2, w)
    
    for k in range(pt1.shape[0]):
        pp = patch1[k]
        pm = torch.isfinite(pp)
        if pm.any():
            m_ = pp[pm].min()
            M_ = pp[pm].max()
        pp[pm] = (pp[pm] - m_) / (M_ - m_)            
        patch1[k] = pp * 255        
    
        pp = patch2[k]
        pm = torch.isfinite(pp)
        if pm.any():
            m_ = pp[pm].min()
            M_ = pp[pm].max()
        pp[pm] = (pp[pm] - m_) / (M_ - m_)            
        patch2[k] = pp * 255       
    
    mask1 = torch.isfinite(patch1) & (~torch.isfinite(patch2))
    patch2[mask1] = 0

    mask2 = torch.isfinite(patch2) & (~torch.isfinite(patch1))
    patch1[mask2] = 0

    both_patches = torch.zeros((3, patch1.shape[0], patch1.shape[1], patch1.shape[2]), dtype=torch.float32, device=device)
    both_patches[0] = patch1
    both_patches[1] = patch2

    save_patch(both_patches, save_prefix=save_prefix, save_suffix='.png', stretch=stretch)


def go_save_list_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_list_diff_', remove_same=False, bar_idx=None, bar_width=2, stretch=False):        
    # warning image must be grayscale and not rgb!

    ww = w * 2 + 1
    l = len(pt1)
    n = pt1[0].shape[0]
    if bar_idx is None: bar_width = 0    
    patch_list = torch.zeros((n, l, ww, bar_width + ww, 3), dtype=torch.float32, device=device)                
    for i in range(l):
        pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1[i], pt2[i], Hs[i]) 
            
        patch1 = patchify(im1, pt1_, Hi1, w)
        patch2 = patchify(im2, pt2_, Hi2, w)
        
        for k in range(n):
            pp = patch1[k]
            pm = torch.isfinite(pp)
            if pm.any():
                m_ = pp[pm].min()
                M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch1[k] = pp * 255        
        
            pp = patch2[k]
            pm = torch.isfinite(pp)
            if pm.any():
                m_ = pp[pm].min()
                M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch2[k] = pp * 255        
    
        mask1 = torch.isfinite(patch1) & (~torch.isfinite(patch2))
        patch2[mask1] = 0

        mask2 = torch.isfinite(patch2) & (~torch.isfinite(patch1))
        patch1[mask2] = 0

        both_patches = torch.zeros((patch1.shape[0], patch1.shape[1], patch1.shape[2] + bar_width, 3), dtype=torch.float32, device=device)
        both_patches[:, :, bar_width:, 0] = patch1
        both_patches[:, :, bar_width:, 1] = patch2        
        
        if not (bar_idx is None):
            patch_idx = torch.zeros((patch1.shape[0], patch1.shape[1], bar_width, 3), dtype=torch.float32, device=device)
            patch_idx[:, :, :, 2] = 128 # more stylish but less visible
            patch_idx[:, :, :] = 255
            for k in range(n):
                ll = torch.clamp(ww - bar_idx[i, k], 0, ww).type(torch.int16)
                # ll = torch.clamp((1 - bar_idx[i, k]) * ww, 0, ww).type(torch.int16)
                patch_idx[k, ll:ww, :, 2] = 255 # more stylish but less visible
                patch_idx[k, ll:ww, :, :2] = 0
        both_patches[:, :, :bar_width, :] = patch_idx

        patch_list[:, i, :, :, :] = both_patches                
            
    if remove_same:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for k in range(n):
            mask[k] = True
            for i in range(l-1):
                if not torch.all(patch_list[k, i, :, :, :].type(torch.uint8) == patch_list[k, i+1, :, :, :].type(torch.uint8)):
                    mask[k] == False
                    break

        patch_list = patch_list[mask]

    patch_list = patch_list.reshape((n, l*ww, ww + bar_width, 3)).permute((-1, 0, 1, 2))

    save_patch(patch_list, grid=[50//l, 50], save_prefix=save_prefix, save_suffix='.png', stretch=stretch)


def go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_', stretch=False):        
    pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1, pt2, Hs) 
            
    patch1 = patchify(im1, pt1_, Hi1, w)
    patch2 = patchify(im2, pt2_, Hi2, w)

    save_patch(patch1, save_prefix=save_prefix, save_suffix='_a.png', stretch=stretch)
    save_patch(patch2, save_prefix=save_prefix, save_suffix='_b.png', stretch=stretch)

    # dx1 = patch1[:, :, 1:-1, :-2] - patch1[:, :, 1:-1, 2:]
    # dy1 = patch1[:, :, :-2, 1:-1] - patch1[:, :, 2:, 1:-1]

    # dx2 = patch2[:, :, 1:-1, :-2] - patch2[:, :, 1:-1, 2:]
    # dy2 = patch2[:, :, :-2, 1:-1] - patch2[:, :, 2:, 1:-1]

    # dm1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
    # dm2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
    
    # dm1 = torch.nn.functional.pad(dm1, (1, 1, 1, 1))
    # dm2 = torch.nn.functional.pad(dm2, (1, 1, 1, 1))
    
    # save_patch(dm1, save_prefix='dm_' + save_prefix, save_suffix='_a.png', stretch=stretch)
    # save_patch(dm2, save_prefix='dm_' + save_prefix, save_suffix='_b.png', stretch=stretch)


def save_patch(patch, grid=[40, 50], save_prefix='patch_', save_suffix='.png', normalize=False, stretch=True):
    if patch.ndim==3:
        patch = patch.unsqueeze(0)
    cc = patch.shape[0]    

    grid_el = grid[0] * grid[1]
    l = patch.shape[1]
    n = patch.shape[2]
    m = patch.shape[3]
    transform = transforms.ToPILImage()
    for i in range(0, l, grid_el):
        j = min(i + grid_el, l)
        filename = f'{save_prefix}{i}_{j}{save_suffix}' 

        if not stretch:
            grid0 = grid[0]
            grid1 = grid[1]
        else:
            grid0 = (j - i) / grid[1]
            if grid0 >= 1:
                grid1 = grid[1]
            else:
                grid1 = (j - i) % grid[1]
            grid0 = int(np.ceil(grid0))
            
        grid_el_ = grid0 * grid1
        
        patch_ = patch[:, i:j]
        aux = torch.zeros((cc, grid_el_, n, m), dtype=torch.float32, device=device)
        aux[:, :j-i] = patch_
        
        mask = aux[0].isfinite()
        aux[:, ~mask] = 0
        
        if not normalize:
            aux = aux.type(torch.uint8)
        else:
            for ci in range(cc):
                aux[ci, ~mask] = -1        
                avg = ((mask * aux[ci]).sum(dim=(1,2)) / mask.sum(dim=(1,2))).reshape(-1, 1, 1).repeat(1, n, m)
                avg[mask] = aux[ci, mask]
                m_ = avg.reshape(grid_el_, -1).min(dim=1)[0]
                M_ = avg.reshape(grid_el_, -1).max(dim=1)[0]
                aux[ci] = (((aux[ci] - m_.reshape(-1, 1, 1)) / (M_ - m_).reshape(-1, 1, 1)) * 255).type(torch.uint8)
           
        # if not needed do not add alpha channel
        all_mask = mask.all()
        
        c_final = cc
        if (~all_mask) and (cc==3): c_final = 4
        if (~all_mask) and (cc==1): c_final = 4

        im = torch.zeros((c_final, grid0 * n, grid1 * m), dtype=torch.uint8, device=device)
        
        aux = aux.reshape(cc, grid0, grid1, n, m).permute(0, 1, 3, 2, 4).reshape(cc, grid0 * n, grid1 * m).contiguous()
        if (~all_mask) and (cc==1): aux = aux.repeat(c_final, 1, 1)
        im[:aux.shape[0]] = aux

        if not all_mask:        
            im[3, :, :] = (mask *255).type(torch.uint8).reshape(grid0, grid1, n, m).permute(0, 2, 1, 3).reshape(grid0 * n, grid1 * m).contiguous()

        transform(im).save(filename)
        

def patchify(img, pts, H, r):

    wi = torch.arange(-r, r+1, device=device)
    ws = r * 2 + 1
    n = pts.size()[0]
    cc, y_sz, x_sz = img.size()
    
    x, y = pts.split(1, dim=1)
    
    widx = torch.zeros((n, 3, ws**2), dtype=torch.float, device=device)
    
    widx[:, 0] = (wi + x).repeat(1, ws)
    widx[:, 1] = (wi + y).repeat_interleave(ws, dim=1)
    widx[:, 2] = 1

    nidx = torch.matmul(H, widx)
    xx, yy, zz = nidx.split(1, dim=1)
    zz_ = zz.squeeze()
    xx_ = xx.squeeze() / zz_
    yy_ = yy.squeeze() / zz_
    
    xf = xx_.floor().type(torch.long)
    yf = yy_.floor().type(torch.long)
    xc = xf + 1
    yc = yf + 1

    nidx_mask = ~torch.isfinite(xx_) | ~torch.isfinite(yy_) | (xf < 0) | (yf < 0) | (xc >= x_sz) | (yc >= y_sz)

    xf[nidx_mask] = 0
    yf[nidx_mask] = 0
    xc[nidx_mask] = 0
    yc[nidx_mask] = 0

    patch = torch.zeros((cc, xx.shape[0],ws,ws), device=device, dtype=torch.float32)    

    for ci in range(cc):
        # for mask
        img_ = img[ci].flatten()
        aux = img_[0].clone()
        img_[0] = float('nan')
    
        a = xx_-xf    
        b = yy_-yf
        c = xc-xx_    
        d = yc-yy_
    
        patch[ci] = (a * (b * img_[yc * x_sz + xc] + d * img_[yf * x_sz + xc]) + c * (b * img_[yc * x_sz + xf] + d * img_[yf * x_sz + xf])).reshape((-1, ws, ws))
        img_[0] = aux

    return patch.squeeze(0)


def dist_shift_table(r=2, d=8, t=True):
    if t:
        i0 = torch.arange(d * 2, device=device) % 2
    else:
        i0 = torch.zeros(d, device=device)
        
    ri = torch.arange(r, device=device).repeat_interleave(d * d)
    di = torch.arange(d, device=device).repeat_interleave(d).repeat(2)
    hi = torch.arange(d, device=device).repeat(r * d)
    
    rri = ri.unsqueeze(0).repeat(d, 1)
    ddi = (di.unsqueeze(0) + torch.arange(8, device=device).unsqueeze(1)) % d
    hhi = (hi.unsqueeze(0) + torch.arange(8, device=device).unsqueeze(1)) % d
    
    i1 = (rri * d * d) + (ddi * d) + hi
    
    return i0, i1


def sgloh_rot(v, a, i0, i1):
    return v[:, i0[a], i1[a // v.shape[1]]]


def sgloh(patch, rad=[12, 20], dirs=8, twice=True, centered=False, hstd=0.7):
    
    rd = rad[-1]
    ws = rd * 2 + 1
    x = torch.arange(-rd, rd + 1).unsqueeze(0).repeat(rd * 2 + 1, 1).to(torch.float)
    y = torch.arange(-rd, rd + 1).unsqueeze(1).repeat(1, rd * 2 + 1).to(torch.float)
    s = 2 * torch.pi / dirs
    
    idx = torch.zeros((1 + twice, ws, ws), device=device)
    idx[0] = ((torch.atan2(y, x) + torch.pi) / s).trunc()
    if twice: idx[1] = ((torch.atan2(y, x) + torch.pi) / s + 0.5).trunc() - 1
    idx[idx == dirs] = 0
    idx[idx < 0] = dirs - 1
    
    d = (x ** 2 + y ** 2) ** 0.5
    r = torch.full((ws, ws), -1, device=device)

    for i in reversed(range(len(rad))):
        mask = d <= rad[i]
        r[mask] = i
        
    if centered:     
        dx = patch[:, 1:-1, :-2] - patch[:, 1:-1, 2:]
        dy = patch[:, :-2, 1:-1] - patch[:, 2:, 1:-1]
    else:
        dx = patch[:, :-2, :-2] - patch[:, :-2, 1:-1]
        dy = patch[:, :-2, :-2] - patch[:, 1:-1, :-2]    
    
    dx[~dx.isfinite()] = 0
    dy[~dy.isfinite()] = 0    
    
    dm = (dx ** 2 + dy ** 2) ** 0.5
    da = (torch.atan2(dy, dx) + torch.pi) / s
    
    desc = torch.zeros((patch.shape[0], idx.shape[0], len(rad), dirs, dirs), device=device)
    
    for k in range(idx.shape[0]):
        for i in range(len(rad)):
            for j in range(dirs):    
                aux_mask = ((r == i) & (idx[k] == j)).unsqueeze(0)
                for w in range(dirs):            
                    v = (da - ((w + j) % dirs)).abs()
                    v[v > dirs / 2] = dirs - v[v > dirs / 2]   
                    desc[:, k, i, j, w] = ((dm * aux_mask) * torch.exp(- (v**2) / (2 * (hstd ** 2)))).sum(dim=[1, 2])
    
    desc = desc.reshape((patch.shape[0], idx.shape[0], -1))
    return desc / desc.sum(dim=-1).unsqueeze(-1)


def prepare_patch(im_gray, pts, H, rad=[12, 20],  im_std=1.0):

    pad = int(np.ceil(3 * im_std))
    
    pts_, H_ = get_inverse(pts, H)
    patch = patchify(im_gray, pts_, H_, rad[-1] + 1 + pad)
    
    if im_std > 0:
        r = np.ceil(3.0 * im_std)   
        g = torch.arange(-r, r+1, device=device)
        s = im_std
        ge = torch.exp(-((g / s) ** 2))
        gn = ge.unsqueeze(1) @ ge.unsqueeze(0)
        gn = gn / gn.sum()
    
        patch = torch.nn.functional.conv2d(patch.unsqueeze(1), gn.unsqueeze(0).unsqueeze(0), padding='valid', ).squeeze(1)

    return patch


def sgloh_dist(d1, d2, to_check=None, pd=1, max_n=1024):
    i0, i1 = dist_shift_table()
      
    vtable = torch.zeros((d1.shape[0], d2.shape[0]))
    itable = torch.zeros((d1.shape[0], d2.shape[0]), dtype=torch.int)
    
    if to_check is None:
        to_check = torch.arange(i0.shape[0])
    
    max_n1 = max_n if np.isfinite(max_n) else d1.shape[0]
    max_n2 = max_n if np.isfinite(max_n) else d2.shape[0]
    
    for i in torch.arange(0, d1.shape[0], max_n1):
        n1 = min(d1.shape[0], i + max_n1)
        d1_shift = sgloh_rot(d1[i:n1], 0, i0, i1)
    
        for j in torch.arange(0, d2.shape[0], max_n2):
            n2 = min(d2.shape[0], j + max_n2)
            
            tmp = torch.zeros((to_check.shape[0], n1 - i, n2 - j))
            for w, k in enumerate(to_check):
                d2_shift = sgloh_rot(d2[j:n2], k, i0, i1)
                tmp[w] = torch.cdist(d1_shift, d2_shift, p=pd)

            val, idx = tmp.min(dim=0)
            vtable[i:n1, j:n2] = val
            itable[i:n1, j:n2] = idx
                        
    return vtable, to_check[itable]


def guess_rot(vtable, itable, i0, to_check=None):
    if to_check is None:
        to_check = torch.arange(i0.shape[0])

    i1 = itable.flatten()[torch.arange(vtable.shape[0]) * vtable.shape[1] + vtable.argmin(dim=1)]
    i2 = itable.flatten()[vtable.argmin(dim=0) * vtable.shape[1] + torch.arange(vtable.shape[1])]

    h = torch.cat((i1, i2)).to(torch.float).histc(bins=i0.shape[0], max=i0.shape[0])

    return to_check[h.argmax()], h


def refined_sgloh_dist(d1, d2, i0, vtable=None, itable=None, rr=2, pd=1, max_n=1024):
    if vtable is None:
        vtable, itable = sgloh_dist(d1, d2, to_check=None, pd=pd, max_n=max_n)
    
    r, h = guess_rot(vtable, itable, i0)
    rot_ref = (torch.arange(-rr, rr + 1) + r) % i0.shape[0]
    
    vtable, itable = sgloh_dist(d1, d2, to_check=rot_ref, pd=pd, max_n=max_n)
    
    return vtable, itable, r, h


def sift(img, lafs, rootsift=True):
    descriptor = cv2.SIFT_create()

    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    kp = opencv_kpts_from_laf(lafs)
    _, desc = descriptor.compute(im, kp)

    if rootsift:
        desc /= desc.sum(axis=1, keepdims=True) + 1e-8
        desc = np.sqrt(desc)
        desc = torch.tensor(desc, device=device, dtype=torch.float)

    return desc
