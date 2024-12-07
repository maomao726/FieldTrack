"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .embedding import EmbeddingComputer
from .association import *
import cv2


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(pos1, pos2):
    cx1, cy1 = pos1[0], pos1[1]
    cx2, cy2 = pos2[0], pos2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, pos, delta_t=3, orig=False, emb = None):
        """
        Initialises a tracker using initial position (x, y)

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=4, dim_z=2)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=4, dim_z=2)
          
        # [x,y,vx,vy]
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])      # state transition matrix
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])      # measurement function

        #self.kf.R[2:, 2:] *= 10.    
        self.kf.P[2:, 2:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        #self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[2:, 2:] *= 0.01

        #self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.kf.x[0], self.kf.x[1] = pos[0], pos[1]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        
        # observation : [x, y, score]
        self.last_observation = np.array([-1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        
        self.emb = emb
        

    def update(self, pos):
        """
        Updates the state vector with observed bbox.
        """
        if pos is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_pos = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_pos = self.observations[self.age-dt]
                        break
                if previous_pos is None:
                    previous_pos = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_pos, pos)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = pos
            self.observations[self.age] = pos
            self.history_observations.append(pos)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            
            self.kf.update(pos[:2].reshape((2, 1)))
        else:
            self.kf.update(None)
    

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if((self.kf.x[6]+self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:2].reshape((1, 2)))
        return self.history[-1] ## return the predicted position, shape: (1, 2)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:2].reshape((1, 2))
    
    def update_emb(self, emb, alpha = 0.9):
        self.emb = (1-alpha) * self.emb + alpha * emb
        self.emb /= np.linalg.norm(self.emb)
    
    def get_emb(self):
        return self.emb


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "mse": mse_batch,
                "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
        dist_threshold=20, mse_tolerence=None, delta_t=3, asso_func="mse", inertia=0.7, 
        use_app_embed=True, weight_app_embed=0.5, app_embed_alpha=0.9,
        use_byte=False):
        """
        Sets key parameters for SORT
        args:
            det_thresh: float, the threshold for detection confidence
            max_age: int, maximum number of missed misses before a track is deleted.
            min_hits: int, minimum number of detections before the track is initialised.

        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.mse_tolerence = mse_tolerence
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        
        self.use_app_embed = use_app_embed
        self.weight_app_embed = weight_app_embed
        self.embedder = EmbeddingComputer(None, None, False)
        self.app_embed_alpha = app_embed_alpha
        KalmanBoxTracker.count = 0



    def update(self, output_results, frame, field_size, warpped_img):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        if output_results is None or output_results[0] is None:
            return np.empty((0, 3))
        

        # get detections (position on field, score, bbox on frame)
        if type(output_results[0]) != np.ndarray:
            proj_obj = output_results[0].cpu().numpy()
            bbox_det = output_results[1].cpu().numpy()
        else:
            proj_obj = output_results[0]
            bbox_det = output_results[1]
        pos_det = proj_obj[:, :2]
        scores = proj_obj[:, 2] * proj_obj[:, 3]

        
        
        #print(dets_emb.shape)

        dets = np.concatenate((pos_det, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        bbox_det = bbox_det[remain_inds]
        
        # generate embeddings
        dets_emb = np.ones((dets.shape[0]))
        if self.use_app_embed:
            dets_emb = self.embedder.compute_embedding(frame, bbox_det, None)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 3))
        trk_emb = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_emb.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # remove invalid rows
        trk_emb = np.array(trk_emb)
        for t in reversed(to_del): 
            self.trackers.pop(t)


        
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        
        debug_img = warpped_img.copy()


        """
            First round of association
        """
        #print(dets.shape, trks.shape, velocities.shape, k_observations.shape)
        #breakpoint()
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, dets_emb, trk_emb, 
            self.dist_threshold, self.mse_tolerence, 
            velocities, k_observations, self.inertia,
            self.use_app_embed, self.weight_app_embed)
        try:
            for m in matched:
                #print(dets[m[0]])
                self.trackers[m[1]].update(dets[m[0], :])
                self.trackers[m[1]].update_emb(dets_emb[m[0]], self.app_embed_alpha)
        except:
            breakpoint()
        

        """
            Second round of associaton by OCR
        """

        # re-association by OCR, high score but unmatched detections
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_emb = dets_emb[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_emb = trk_emb[unmatched_trks]
            
            # TODO : maybe also use app_embed here?
            dist_left = self.asso_func(left_dets, left_trks, self.dist_threshold, self.mse_tolerence)
            dist_left = np.array(dist_left)
            if dist_left.max() > 0:
                rematched_indices = linear_assignment(-dist_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if dist_left[m[0], m[1]] <= 0:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    self.trackers[trk_ind].update_emb(dets_emb[det_ind], self.app_embed_alpha)
                    
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        """
            Third round of association by ByteTrack
        """
        # BYTE association, close low score detections to unmatched tracks
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            dist_left = self.asso_func(dets_second, u_trks, self.dist_threshold, self.mse_tolerence)          # iou between low score detections and unmatched tracks
            dist_left = np.array(dist_left)
            
            if dist_left.max() > 0:
                matched_indices = linear_assignment(-dist_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if dist_left[m[0], m[1]] <= 0:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t, emb=dets_emb[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:2]
            #print(trk.get_state(), trk.last_observation[:2])
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if(len(ret) > 0):
            return np.concatenate(ret)
        
        
        return np.empty((0, 3))

