import cv2
import threading
import queue
import numpy as np
from tqdm import tqdm


class Stabilizator:
    def __init__(self, input_video_path: str, scale: float, step: int):
        """
        Video stabilizer for long and heavy weighed videos.
        This function stabilizes videos efficiently by using downscaling and calculating warp matrices only for keyframes at a specified interval (step).
            input_video_path (str) - initial video path,
            scale (float) - downscaling value used when calculating warp matrices. Smaller values speed up calculations but may reduce stabilization quality if set too low
            step (int) - iterval between keyframes for warp matrix calculation. For highly shaky videos, use a smaller step. For videos with minor camera movements, a larger step can be used for faster processing.
        """
        self.input_video_path = input_video_path
        self.scale = scale
        self.step = step

        # reading video information
        cap = cv2.VideoCapture(self.input_video_path)
        ret, frame = cap.read()
        if not ret:
            print('Cant read videofile')
            exit()
        self.frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.init_image_shape = frame.shape
        cap.release()

        self.warp_stack = []
        self.warp_stack_decomposed = []

        # initialize keyframes
        self.key_frames = list(range(self.step, self.frames_number, self.step))

        # processing last frame separately
        if (self.frames_number - 1) % self.step != 0:
            self.key_frames += [self.frames_number - 1]

    def get_warp(self, img1, img2, motion = cv2.MOTION_EUCLIDEAN):
        """Calculating warp matrix for two frames. For now Works with cv2.MOTION_EUCLIDEAN only"""
        imga = img1.copy().astype(np.float32)
        imgb = img2.copy().astype(np.float32)
        if len(imga.shape) == 3:
            imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        if len(imgb.shape) == 3:
            imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
        warpMatrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,
                                        warpMatrix=warpMatrix, motionType=motion)[1]
        return warp_matrix
    
    def calculate_warp_stack(self):
        """Calculate warp matrics for frames with a given step"""
        cap = cv2.VideoCapture(self.input_video_path)
        ret, prev_frame = cap.read()
        if not ret:
            print('Cant read videofile')
            exit()

        self.warp_stack = [np.eye(2, 3)]

        prev_frame = cv2.resize(prev_frame, None, fx=self.scale, fy=self.scale)
        
        for key_frame_idx in tqdm(self.key_frames, desc="Processing key-frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame_idx - 1)
            ret, cur_frame = cap.read()
            if not ret:
                print('Cant read videofile')
                break
            cur_frame = cv2.resize(cur_frame, None, fx=self.scale, fy=self.scale)
            self.warp_stack += [self.get_warp(prev_frame, cur_frame)]
            prev_frame = cur_frame

        self.warp_stack = np.array(self.warp_stack)
        cap.release()

    
    def warp_stack_decomposition(self):
        """
        Decomposes warp matrices of MOTION_EUCLIDEAN on theta angle and x y translations.
        Distibutes warps of key frame by every single frame of video file.
        """
        # Decomposition
        thetas = np.arctan2(self.warp_stack[:, 1, 0], self.warp_stack[:, 0, 0])
        translations = self.warp_stack[:, :, 2]
        # distribution by frames
        # Processing last frame separately
        if self.key_frames[-1] % self.step  == 0:
            theta_distributed = np.repeat(thetas[1:] / self.step , self.step)
            tx_distributed = np.repeat(translations[1:, 0] / self.step , self.step)
            ty_distributed = np.repeat(translations[1:, 1] / self.step , self.step)
        else:
            last_step = self.key_frames[-1] % self.step 
            theta_distributed = np.concatenate((np.repeat(thetas[1: -1] / self.step , self.step ), np.repeat(thetas[-1] / last_step, last_step)))
            tx_distributed = np.concatenate((np.repeat(translations[1: -1, 0] / self.step , self.step ), np.repeat(translations[-1, 0] / last_step, last_step)))
            ty_distributed = np.concatenate((np.repeat(translations[1: -1, 1] / self.step , self.step ), np.repeat(translations[-1, 1] / last_step, last_step)))
        
        cos_theta_distributed = np.cos(theta_distributed)
        sin_theta_distributed = np.sin(theta_distributed)
        self.warp_stack_decomposed = np.array([
                [cos_theta_distributed, -sin_theta_distributed, tx_distributed],
                [sin_theta_distributed, cos_theta_distributed, ty_distributed]
            ], dtype=np.float32).transpose(2, 0, 1)
        
            
    def _get_border_pads(self):
        """Calculates max borders for stabilization"""
        maxmin = []
        corners = np.array([[0,0,1], [self.init_image_shape[1], 0, 1], [0, self.init_image_shape[0],1], [self.init_image_shape[1], self.init_image_shape[0], 1]]).T
        warp_prev = np.eye(3)
        for warp in self.warp_stack:
            warp = np.concatenate([warp, [[0,0,1]]])
            warp = np.matmul(warp, warp_prev)
            warp_invs = np.linalg.inv(warp)
            new_corners = np.matmul(warp_invs, corners)
            xmax,xmin = new_corners[0].max(), new_corners[0].min()
            ymax,ymin = new_corners[1].max(), new_corners[1].min()
            maxmin += [[ymax,xmax], [ymin,xmin]]
            warp_prev = warp.copy()
        maxmin = np.array(maxmin)
        bottom = maxmin[:,0].max()
        print('bottom', maxmin[:,0].argmax()//2)
        top = maxmin[:,0].min()
        print('top', maxmin[:,0].argmin()//2)
        left = maxmin[:,1].min()
        print('right', maxmin[:,1].argmax()//2)
        right = maxmin[:,1].max()
        print('left', maxmin[:,1].argmin()//2)
        return int(-top), int(bottom - self.init_image_shape[0]), int(-left), int(right - self.init_image_shape[1])
    

    def warp_rescaling(self, warp_stack):
        """Uppscaling wrap back to initial image"""
        scaled_warp_stack = warp_stack.copy()
        scaled_warp_stack[:, :, 2] /= self.scale
        return scaled_warp_stack
    
    def homography_gen(self, warp_stack):
        """Calculates homography translation"""
        H_tot = np.eye(3)
        wsp = np.dstack([warp_stack[:, 0, :], warp_stack[:, 1, :], np.array([[0, 0, 1]] * warp_stack.shape[0])])
        for i in range(len(warp_stack)):
            H_tot = np.matmul(wsp[i].T, H_tot)
            yield np.linalg.inv(H_tot)

    def apply_warping_fullview(self, save_path: str = None, show: bool = False):
        """
        Stabilizes video by applying precomputed warp matrices to each frame. 
        Optionally saves the output to a file and/or displays it in real time. Uses threading for efficient video writing.
        """

        def write_frames_to_video(cap_out, processed_queue):
            while True:
                frame = processed_queue.get()
                if frame is None:
                    break
                cap_out.write(frame)

        cap = cv2.VideoCapture(self.input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, prev_frame = cap.read()
        if not ret:
            print('Cant read videofile')
            exit()

        warp_stack = self.warp_rescaling(self.warp_stack_decomposed)
        top, bottom, left, right = self._get_border_pads(warp_stack=warp_stack)
        width = self.init_image_shape[1] + left + right
        height =  self.init_image_shape[0] + top + bottom
        H = self.homography_gen(warp_stack)


        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            cap_out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            write_video_queue = queue.Queue(maxsize=10)
            write_video_thread = threading.Thread(target=write_frames_to_video, args=(cap_out, write_video_queue))
            write_video_thread.daemon = True  # Поток записи работает в фоновом режиме
            write_video_thread.start()
            write_video_queue.put(prev_frame)

            

        with tqdm(total=self.frames_number, initial=1, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('End of videofile')
                    break

                H_tot = next(H) + np.array([[0, 0, left], [0, 0, top], [0, 0, 0]])
                frame_warp = cv2.warpPerspective(frame, H_tot, (width, height))
                if save_path:
                    write_video_queue.put(frame_warp)
                    
                if show:
                    cv2.imshow('Stabilized video', frame_warp)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break

                pbar.update(1)

        cap.release()
        if save_path:
            write_video_queue.put(None)
            write_video_thread.join()
            cap_out.release()
        if show:
            cv2.destroyAllWindows()