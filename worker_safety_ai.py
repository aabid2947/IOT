import yolov5
import cv2
import numpy

def video(img):
    video=cv2.VideoWriter('video15.avi',cv2.VideoWriter_fourcc(*'mp4v'), 20,(200,200))

    count = 0
    for image in img:
        cv2.imshow("image",image)
        cv2.imwrite(f"./output/img{count}",image)
        video.write(image)
        count =+1


def frame(path,model):

   # Path to video file 
    vidObj = cv2.VideoCapture(path) 

    succes , image = vidObj.read()
    
    # count the number of frames
    count = 0

      # checks whether frames were extracted 
    success = 1
    print(vidObj.get(cv2.CAP_PROP_FPS))
    
    img =[]

    while succes:
    # while vidObj.isOpened():

        success , image = vidObj.read()
        size =  image.shape

        if success :
       
          # perform inference
          results = model(image, size=640)

          # inference with test time augmentation
          results = model(image, augment=True)

          # parse results
          predictions = results.pred[0]
          boxes = predictions[:, :4] # x1, y1, x2, y2
          scores = predictions[:, 4]
          categories = predictions[:, 5]
        


          # show detection bounding boxes on image
          # results.show()

          img.append(numpy.squeeze(results.render()))
          cv2.imshow("detection",numpy.squeeze(results.render()))
          if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
      
          
          # plt.imshow(numpy.squeeze(results.render()))
          # plt.show()
          # print(results.render())

          count +=1
    cv2.destroyAllWindows()
    # video(img=img)
    # cv2.imshow("image",img[0])
    # video(img)
    
   
    


def main():
    # load model
    model = yolov5.load('keremberke/yolov5n-construction-safety')
    
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    frame('./27.03.2024_16.43.00_REC.mp4',model=model)





main()