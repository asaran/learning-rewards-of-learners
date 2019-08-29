import re
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DatasetWithHeatmap:
    def __init__(self):
        # train_imgs, train_lbl, train_fid, train_size, train_weight = None, None, None, None, None
        self.frameid2pos = None
        self.GHmap = None # GHmap means gaze heap map
        self.NUM_ACTION = 18
        self.xSCALE, self.ySCALE = 8, 4 # was 6,3
        self.SCR_W, self.SCR_H = 160*self.xSCALE, 210*self.ySCALE
        self.train_size = 10
        self.HEATMAP_SHAPE = 14
        
    def createGazeHeatmap(self, gaze_coords, heatmap_shape):
        # print("Reading gaze data ASC file, and converting per-frame gaze positions to heat map...")
        # print(gaze_coords)
        self.frameid2pos = self.get_gaze_data(gaze_coords)
        # print(self.frameid2pos)
        self.train_size = len(self.frameid2pos.keys())
        self.HEATMAP_SHAPE = heatmap_shape
    
        if(heatmap_shape<=7):
            self.HEATMAP_SHAPE *= 2
        self.GHmap = np.zeros([self.train_size, self.HEATMAP_SHAPE, self.HEATMAP_SHAPE, 1], dtype=np.float32)
        
        # print("Running BIU.convert_gaze_pos_to_heap_map() and convolution...")
        t1 = time.time()
        
        bad_count, tot_count = 0, 0
        for (i,fid) in enumerate(self.frameid2pos.keys()):
            tot_count += len(self.frameid2pos[fid])
            bad_count += self.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.GHmap[i])
            
        # print("Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count))    
        # print("'Bad' means the gaze position is outside the 160*210 screen")
        
        sigmaH = 28.50 * self.HEATMAP_SHAPE / self.SCR_H
        sigmaW = 44.58 * self.HEATMAP_SHAPE / self.SCR_W
        self.GHmap = self.preprocess_gaze_heatmap(sigmaH, sigmaW, 0).astype(np.float32)
        # print(np.count_nonzero(self.GHmap))

        if heatmap_shape<=7:
            import scipy.ndimage
            self.GHmap = scipy.ndimage.zoom(self.GHmap, (1, 0.5, 0.5, 1))

        # print("Normalizing the heat map...")
        for i in range(len(self.GHmap)):
            SUM = self.GHmap[i].sum()
            if SUM != 0:
                self.GHmap[i] /= SUM

        # print("Done. BIU.convert_gaze_pos_to_heap_map() and convolution used: %.1fs" % (time.time()-t1))
        # print(type(self.GHmap))
        if not np.count_nonzero(self.GHmap):
            # print(gaze_coords)
            print('The gaze map is all zeros')
            

        return self.GHmap
    
    def get_gaze_data(self, gaze_coords):
        frameid2pos = {}
        frame_id = 0
        for gaze_list in gaze_coords:
            frameid2pos[frame_id] = gaze_list
            frame_id += 1    

        # if len(frameid2pos) < 1000: # simple sanity check
        #     print ("Warning: did you provide the correct gaze data? Because the data for only %d frames is detected" % (len(frameid2pos)))

        few_cnt = 0
        for v in frameid2pos.values():
            if len(v) < 10: few_cnt += 1
        # print ("Warning:  %d frames have less than 10 gaze samples. (%.1f%%, total frame: %d)" % \
            # (few_cnt, 100.0*few_cnt/len(frameid2pos), len(frameid2pos)))     

        return frameid2pos

    
    # bg_prob_density seems to hurt accuracy. Better set it to 0
    def preprocess_gaze_heatmap(self, sigmaH, sigmaW, bg_prob_density, debug_plot_result=False):
        from scipy.stats import multivariate_normal
        import tensorflow as tf, keras as K # don't move this to the top, as people who import this file might not have keras or tf
#         print(self.GHmap[0,:,:,:])
        model = K.models.Sequential()
        model.add(K.layers.Lambda(lambda x: x+bg_prob_density, input_shape=(self.GHmap.shape[1],self.GHmap.shape[2],1)))

        if sigmaH > 0.0 and sigmaW > 0.0:
            lh, lw = int(4*sigmaH), int(4*sigmaW)
            x, y = np.mgrid[-lh:lh+1:1, -lw:lw+1:1] # so the kernel size is [lh*2+1,lw*2+1]
            pos = np.dstack((x, y))
            gkernel=multivariate_normal.pdf(pos,mean=[0,0],cov=[[sigmaH*sigmaH,0],[0,sigmaW*sigmaW]])
            assert gkernel.sum() > 0.95, "Simple sanity check: prob density should add up to nearly 1.0"

            model.add(K.layers.Lambda(lambda x: tf.pad(x,[(0,0),(lh,lh),(lw,lw),(0,0)],'REFLECT')))
            # print(gkernel.shape, sigmaH, sigmaW)
            model.add(K.layers.Conv2D(1, kernel_size=gkernel.shape, strides=1, padding="valid", use_bias=False,
                activation="linear", kernel_initializer=K.initializers.Constant(gkernel)))
        else:
            print ("WARNING: Gaussian filter's sigma is 0, i.e. no blur.")
        # The following normalization hurts accuracy. I don't know why. But intuitively it should increase accuracy
        #def GH_normalization(x):
        #    sum_per_GH = tf.reduce_sum(x,axis=[1,2,3])
        #    sum_per_GH_correct_shape = tf.reshape(sum_per_GH, [tf.shape(sum_per_GH)[0],1,1,1])
        #    # normalize values to range [0,1], on a per heap-map basis
        #    x = x/sum_per_GH_correct_shape
        #    return x
        #model.add(K.layers.Lambda(lambda x: GH_normalization(x)))
        
        model.compile(optimizer='rmsprop', # not used
            loss='categorical_crossentropy', # not used
            metrics=None)
        
        # print(np.count_nonzero(self.GHmap))
        output=model.predict(self.GHmap, batch_size=500)
        # print(type(output))
        # print(np.count_nonzero(output))

        if debug_plot_result:
            print (r"""debug_plot_result is True. Entering IPython console. You can run:
                    %matplotlib
                    import matplotlib.pyplot as plt
                    f, axarr = plt.subplots(1,2)
                    axarr[0].imshow(gkernel)
                    rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
                    axarr[1].imshow(output[rnd,...,0])""")
            embed()
        
        shape_before, shape_after = self.GHmap.shape, output.shape
        assert shape_before == shape_after, """
        Simple sanity check: shape changed after preprocessing. 
        Your preprocessing code might be wrong. Check the shape of output tensor of your tensorflow code above"""
        return output
    
    def make_unique_frame_id(self, UTID, frameid):
        return (hash(UTID), int(frameid))
    
    def convert_gaze_pos_to_heap_map(self, gaze_pos_list, out):
        h,w = out.shape[0], out.shape[1]
        bad_count = 0
        # for (x,y) in gaze_pos_list: 
        # print(gaze_pos_list)
        # print(len(gaze_pos_list))
        # if(not np.isnan(gaze_pos_list).all()): 
        if not np.isnan(gaze_pos_list).all():
            for j in range(0,len(gaze_pos_list),2):
                x = gaze_pos_list[j]
                y = gaze_pos_list[j+1]
                try:
                    out[int(y/self.SCR_H*h), int(x/self.SCR_W*w)] += 1
                except IndexError: # the computed X,Y position is not in the gaze heat map
                    bad_count += 1
        return bad_count