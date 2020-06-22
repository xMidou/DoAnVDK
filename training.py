

# Import OpenCV2 for image processing
# Import os for file path
import cv2, os

# Import numpy for matrix calculation -> thư viện numpy
import numpy as np

# Import Python Image Library (PIL) -> thư viện PIL
from PIL import Image

# Create Local Binary Patterns Histograms for face recognization -> tạo biểu đồ tần suất Local Binary Patterns cho nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection -> sử dụng model build trước frontal face traning 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create method to get the images and label data -> tạo phương thức để lấy ảnh và nhãn data
def getImagesAndLabels(path):
    
    # Get all file path -> lấy đường dẫn file
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample -> bắt đầu ?
    faceSamples=[]
    
    # Initialize empty id ->
    ids = []
    count = 0
    # Loop all the file path -> lặp với mỗi đường dẫn
    for imagePath in imagePaths:
        try:
            # Get the image and convert it to grayscale -> lấy hình ảnh và chuyển thành grayscale
            PIL_img = Image.open(imagePath).convert('L')

            # PIL image to numpy array -> ảnh PIL -> mảng numpy
            img_numpy = np.array(PIL_img,'uint8')
            
            # Get the image id -> lấy id ảnh
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            print(str(id) + str(count))
            count += 1;
            # Get the face from the training images -> lấy khuôn mặt từ ảnh đang train
            faces = detector.detectMultiScale(img_numpy)

            # Loop for each face, append to their respective ID -> 
            for (x,y,w,h) in faces:

                # Add the image to face samples -> thêm ảnh vào mẫu
                faceSamples.append(img_numpy[y:y+h,x:x+w])

                # Add the ID to IDs -> thêm id 
                ids.append(id)
        except:
            print('error')
    # Pass the face array and IDs array -> trả về 
    return faceSamples,ids

# Get the faces and IDs -> 
faces,ids = getImagesAndLabels('dataset')

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.save('trainer/trainer.yml')
