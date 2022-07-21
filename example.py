from api import Detector
import time

t1 = time.time_ns()
# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                    use_cuda=False)
t1 = time.time_ns() - t1

t2 = time.time_ns()
# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./images/exhibition.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=False)
t2 = time.time_ns() - t2

t3 = time.time_ns()
# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                    use_cuda=True)
t3 = time.time_ns() - t3

t4 = time.time_ns()
# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./images/exhibition.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=False)
t4 = time.time_ns() - t4

print("Time for initialize(CPU): " + str(t1) + "ns (" + str(t1/1000000000) +"s)")
print("Time for detecting(CPU): " + str(t2) + "ns (" + str(t2/1000000000) +"s)")
print("Time for initialize(CUDA): " + str(t3) + "ns (" + str(t3/1000000000) +"s)")
print("Time for detecting(CUDA): " + str(t4) + "ns (" + str(t4/1000000000) +"s)")