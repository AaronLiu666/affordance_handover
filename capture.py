import pyrealsense2 as rs
import numpy as np
import cv2
from pynput import keyboard

# global variable indicating image start index
image_index = 11
save_dir = 'test_data/layout_test'

class RealSenseCamera:
    def __init__(self):
        self.pipeline, self.config = self.configure_device()
        self.color_image = None
        self.running = True  # Add a running flag
        self.stream_started = False  # Add a flag to track if streaming has started

    def configure_device(self):
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        return pipeline, config

    def start_streaming(self):
        try:
            self.pipeline.start(self.config)
            self.stream_started = True
        except Exception as e:
            print(f"Failed to start pipeline: {e}")
            exit(1)

    def on_press(self, key):
        try:
            if key.char == 'c' and self.color_image is not None:
                # image_name increases from image_index by 1 for each capture
                global image_index
                # image_name is 'image_<image_index>.png', and image_index format as 4 digits
                image_name = 'image_{}.png'.format(str(image_index).zfill(4))
                cv2.imwrite(save_dir+'/'+image_name, self.color_image)
                print('saved ' + image_name)
                image_index += 1
            elif key.char == 'q':
                self.running = False
                self.stop_streaming()
                return False
        except AttributeError:
            pass

    def start_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        return listener

    def process_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as e:
            print(f"Failed to get frames: {e}")
            return None

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = self.color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((self.color_image, depth_colormap))

        return images

    def display_images(self, images):
        if images is not None:
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    def stop_streaming(self):
        if self.stream_started:
            self.pipeline.stop()
            self.stream_started = False
        cv2.destroyAllWindows()

    def run(self):
        self.start_streaming()
        listener = self.start_listener()
        try:
            while self.running:
                images = self.process_frames()
                if images is not None:
                    self.display_images(images)
        finally:
            listener.stop()
            if self.stream_started:
                self.stop_streaming()

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.run()