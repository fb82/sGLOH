import sys
import os
import torch
import cv2
import poselib
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_kpts
import numpy as np
import DTM.src.dtm as dtm
import DTM.hz.hz as hz
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms as transforms
from PIL import Image
import src.sgloh as sgloh

if __name__ == "__main__":
    # device to use (with the exception of Blob Matching running always on CPU to avoid OOM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if you need, change to
    # device = 'cpu'
   
    warnings.warn("if your GPU has little amount of memory (i.e. 4GB), set device='cpu' or change the keypoint detectors or the image pair as detailed in this demo.py code")    
    
    im_pair = ['DTM/data/DC/dc0.png', 'DTM/data/DC/dc2.png']
    # or if you have low amount of GPU memory
    # im_pair = ['DTM/data/ET/et000.jpg', 'DTM/data/ET/et001.jpg']

    # you can give non-default input image pair as further arguments 
    if len(sys.argv) >= 3: img = [sys.argv[1], sys.argv[2]]
    else: img = im_pair

    # more keypoint detectors (or just one if you don't have enough memory)
    detectors = {
        'Hz+': 9,  # patch rescaling, -1 for none
        'DoG': 12, # patch rescaling, -1 for none
        }

    # rotation mode is  sGOr2a
    rot_mode = {'refine around dominant'} 
    # or sGLOH2
    # rot_mode = {'best patch match'}
    # or upright with a fixed orientation
    # rot_mode = {'fixed': 1} # upright with fixed discrete rotation in [0, 15]
    # or just use SIFT for comparison
    # rot_mode = {'SIFT'}
    
    # matching
    matcher = {'Blob Matching'}
    # or just
    # matcher = {'Mutual Nearest Neighbor (MNN)': 0.9}
    
    # in case one does not want to use dissimilarity values of matches but only the spatial keypoint localization on the images
    dtm_only_spatial = False
    
    # Delaunay pre-quantization to redeuce the spatial grid resolution
    # kp = (round(kp * s + t) - t) / s
    dtm_st = [1., 0.] 
    
    # visualize DTM steps
    dtm_show_in_progress = False

    # DTM border handling
    # the original Matlab approach is not implemented due to laziness (the matlab boundary function does not exist in Python)
    # but this should work similar
    dtm_prepare_data = dtm.prepare_data_shaped         

    # RANSAC fundamental matrix estimation parameters    
    poselib_params = {            
     'max_iterations': 100000,
     'min_iterations': 50,
     'success_prob': 0.9999,
     'max_epipolar_error': 3,
     }    
    
    # guided matching iterations, done by forcing their similarity values to the lowest value    
    # currently not done
    ii = 0
    # for doing once set to 
    # ii = 1
    
    if (not os.path.isfile(img[0])) or (not os.path.isfile(img[1])):
        print('one or both input images not found!')
    
    with torch.no_grad():        
        laf0 = torch.zeros((1, 0, 2, 3), device=device, dtype=torch.float)        
        laf1 = torch.zeros((1, 0, 2, 3), device=device, dtype=torch.float)        
        scale0 = torch.zeros((0), device=device, dtype=torch.float)
        scale1 = torch.zeros((0), device=device, dtype=torch.float)

        # Hz+
        if 'Hz+' in detectors:
            hz0, _ = hz.hz_plus(hz.load_to_tensor(img[0]).to(torch.float), output_format='laf')
            hz0 = KF.ellipse_to_laf(hz0[None]).to(device).to(torch.float)
            laf0 = torch.concat((laf0, hz0), dim=1)
            scale0 = torch.concat((scale0, torch.full((hz0.shape[1], ), 1 / detectors['Hz+'], device=device)))

            hz1, _ = hz.hz_plus(hz.load_to_tensor(img[1]).to(torch.float), output_format='laf')
            hz1 = KF.ellipse_to_laf(hz1[None]).to(device).to(torch.float)
            laf1 = torch.concat((laf1, hz1), dim=1)
            scale1 = torch.concat((scale1, torch.full((hz1.shape[1], ), 1 / detectors['Hz+'], device=device)))

        # DoG
        if 'DoG' in detectors:        
            dog = cv2.SIFT_create(nfeatures=8000, contrastThreshold=-10000, edgeThreshold=10000)

            dog0 = laf_from_opencv_kpts(dog.detect(cv2.imread(img[0], cv2.IMREAD_GRAYSCALE), None), device=device).to(torch.float)
            laf0 = torch.concat((laf0, dog0), dim=1)
            scale0 = torch.concat((scale0, torch.full((dog0.shape[1], ), 1 / detectors['DoG'], device=device)))

            dog1 = laf_from_opencv_kpts(dog.detect(cv2.imread(img[1], cv2.IMREAD_GRAYSCALE), None), device=device).to(torch.float)
            laf1 = torch.concat((laf1, dog1), dim=1)
            scale1 = torch.concat((scale1, torch.full((dog1.shape[1], ), 1 / detectors['DoG'], device=device)))
        
        # grayscale image load for sGLOH
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 
                
        transform_none = transforms.PILToTensor() 
        
        im0 = Image.open(img[0])
        timg0 = transform(im0).type(torch.float16).to(device)

        im1 = Image.open(img[1])
        timg1 = transform(im1).type(torch.float16).to(device)

        # sGLOH descriptor & matching table
        i0, i1 = sgloh.dist_shift_table()

        if 'SIFT' in rot_mode:
            # SIFT descriptors, for comparison
            desc0_base = sgloh.sift(img[0], laf0)
            desc1_base = sgloh.sift(img[1], laf1)

            kp0, _, _ = sgloh.laf2homo(laf0.squeeze(0))
            kp1, _, _ = sgloh.laf2homo(laf1.squeeze(0))

            vtable = None
        else:
            # make upright
            laf0 = KF.set_laf_orientation(laf0, torch.zeros((laf0.shape[0], laf0.shape[1], 1), device=device))
            kp0, H0, s0 = sgloh.laf2homo(laf0.squeeze(0))
            # remerge homography and scale
            Hs0 = H0
            scale0[scale0 < 1] = 1 / s0[scale0 < 1]
            Hs0[:, :2, :] = Hs0[:, :2, :] * (s0 * scale0).unsqueeze(1).unsqueeze(1) 
            patch0 = sgloh.prepare_patch(timg0, kp0, Hs0)
            desc0 = sgloh.sgloh(patch0)
            # save patches for visualization
            sgloh.save_patch(patch0, grid=[50, 50], save_prefix='patch0_', save_suffix='.png', normalize=False, stretch=True)
    
            # make upright
            laf1 = KF.set_laf_orientation(laf1, torch.zeros((laf1.shape[0], laf1.shape[1], 1), device=device))        
            kp1, H1, s1 = sgloh.laf2homo(laf1.squeeze(0))
            # remerge homography and scale
            Hs1 = H1
            scale1[scale1 < 1] = 1 / s1[scale1 < 1]
            Hs1[:, :2, :] = Hs1[:, :2, :] * (s1 * scale1).unsqueeze(1).unsqueeze(1)
            patch1 = sgloh.prepare_patch(timg1, kp1, Hs1)
            desc1 = sgloh.sgloh(patch1)
            # save patches for visualization
            sgloh.save_patch(patch1, grid=[50, 50], save_prefix='patch1_', save_suffix='.png', normalize=False, stretch=True)
           
            if 'refine around dominant' in rot_mode:
                # guess the main orientation and then check for the refined match
                vtable, itable, rot, h = sgloh.refined_sgloh_dist(desc0, desc1, i0)
            elif 'best patch match' in rot_mode:
                # or just take for each patch rotation the one that minimize the match
                vtable, itable = sgloh.sgloh_dist(desc0, desc1)
            else:
                # or check matches for a fixed orientation
                vtable = None
                desc0 = sgloh.sgloh(patch0)
                desc1 = sgloh.sgloh(patch1)
                desc0_base = sgloh.sgloh_rot(desc0, 0, i0, i1)
                desc1_base = sgloh.sgloh_rot(desc1, rot_mode['fixed'], i0, i1)
            
        if 'Blob Matching' in matcher:
            # Blob matching (on CPU to avoid OOM)
            # note the m=vtable parameter

            if ('refine around dominant' in rot_mode) or ('best patch match' in rot_mode):
                # actually is dummy
                desc0_base = None
                desc1_base = None

            m_idx, m_val = dtm.blob_matching(kp0, kp1, desc0_base, desc1_base, device='cpu', m=vtable)
            m_idx = m_idx.to(device)
            m_val = m_val.to(device)
            m_mask = torch.ones(m_val.shape[0], device=device, dtype=torch.bool)
        else:
            # Mutual NN matching (with a high threshold)
            # note the dm=vtable parameter

            if ('refine around dominant' in rot_mode) or ('best patch match' in rot_mode):
                # actually is dummy
                desc0 = sgloh.sgloh(patch0)
                desc1 = sgloh.sgloh(patch1)
                desc0_base = sgloh.sgloh_rot(desc0, 0, i0, i1)
                desc1_base = sgloh.sgloh_rot(desc1, 0, i0, i1)

            th = matcher['Mutual Nearest Neighbor (MNN)']
            m_val, m_idx = K.feature.match_smnn(desc0_base, desc1_base, th, dm=vtable)
            m_val = m_val.squeeze(1).to(device)
            m_idx = m_idx.to(device)
            m_mask = torch.ones(m_val.shape[0], device=device, dtype=torch.bool)
    
        # DTM
        match_data = {
            'img': img,
            'kp': [kp0, kp1],
            'm_idx': m_idx,
            'm_val': m_val,
            'm_mask': m_mask,
            }

        # if one just wants to use spatial clues only and not similarity    
        if dtm_only_spatial: match_data['m_val'][:] = 1.0

        # retained matches are signed values <= 0 in the mask
        # values > 0 indicate at which iteration the match was discarded
        # negative values indicate the iteraion in the 2nd step the match was re-included  
        dtm_mask = dtm.dtm(match_data, show_in_progress=dtm_show_in_progress, prepare_data=dtm_prepare_data) <= 0

        # RANSAC         
        idx = m_idx.to('cpu').detach()
        pt0 = np.ascontiguousarray(kp0.to('cpu').detach())[idx[:, 0]]
        pt1 = np.ascontiguousarray(kp1.to('cpu').detach())[idx[:, 1]]   
        
        F, info = poselib.estimate_fundamental(pt0[dtm_mask], pt1[dtm_mask], poselib_params, {})
        poselib_mask = info['inliers']
        sac_mask = np.copy(dtm_mask)
        sac_mask[dtm_mask] = poselib_mask
        
        # show matches (those discarded by ransac are in red)
        dtm.plot_pair_matches(img, pt0, pt1, dtm_mask, sac_mask)
         
        # the guided filtering can be done zero or more times
        for i in range(ii):
            # re-filter with DTM, guided filtering on previous matches by forcing their similarity values to the lowest value
            match_data['m_val'][sac_mask] = 0
            dtm_mask = dtm.dtm(match_data, show_in_progress=dtm_show_in_progress, prepare_data=dtm_prepare_data) <= 0
        
            # RANSAC on re-filtered matches
            idx = m_idx.to('cpu').detach()
            pt0 = np.ascontiguousarray(kp0.to('cpu').detach())[idx[:, 0]]
            pt1 = np.ascontiguousarray(kp1.to('cpu').detach())[idx[:, 1]]   
            
            F, info = poselib.estimate_fundamental(pt0[dtm_mask], pt1[dtm_mask], poselib_params, {})
            poselib_mask = info['inliers']
            sac_mask = np.copy(dtm_mask)
            sac_mask[dtm_mask] = poselib_mask
        
            # show matches
            dtm.plot_pair_matches(img, pt0, pt1, dtm_mask, sac_mask)
            
        # force plots to show if not already happened
        plt.show()
