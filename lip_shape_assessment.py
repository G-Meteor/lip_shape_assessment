import numpy as np
import pandas as pd
import time
import os
from posixpath import basename
import cv2
import mediapipe as mp
import math
from moviepy.editor import AudioFileClip
import re


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def distance(point1, point2):
    dis = math.sqrt((point1[0]-point2[0])*(point1[0] - point2[0]) +
                    (point1[1]-point2[1])*(point1[1] - point2[1]))
    return dis


def mediapipe478_251(me: list):
    # 478关键点坐标转为251点坐标
    try:
        r251 = [(x*0, x*0) for x in range(0, 251)]
        # print(len(r251))
        data = pd.read_excel('convert.xls')
        for row in data.values:
            # print(row)
            id_251 = int(row[1])
            m1 = int(row[2])
            m2 = row[3]
            ratio = row[4]

            if m2 != np.nan:  # is nan
                r251[id_251] = me[m1]
            else:
                # m2 is not nan
                m2 = int(row[3])
                x1 = me[m1][0]
                y1 = me[m1][1]
                x2 = me[m2][0]
                y2 = me[m2][1]
                r251[id_251] = (int(x1+(x2-x1)*ratio),
                                int(y1+(y2-y1)*ratio))
        # print(r251)
        return r251
    except:
        print('handle ')


def webcam_input():
    # 视频流输入（需更改）
    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture('访谈.mp4')
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    faceLms = face_landmarks
                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic = image.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        print(id, x, y)
                        cv2.putText(image, str(id), (x, y),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                    # 嘴唇
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # 左眼
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    # 做眉毛
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # FACEMESH_RIGHT_EYE
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # FACEMESH_RIGHT_EYE
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            cv2.imshow("mediapipe", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def generate_478points_img(image):
    # 生成图片流中478个人脸关键点
    # For static images:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        # h, w = image.shape[0], image.shape[1]
        # image = cv2.resize(image, (int(w*8), int(h*8)), interpolation=cv2.INTER_LINEAR)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return []

        # annotated_image = image.copy()
        landmark = []
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            faceLms = face_landmarks
            # print(len(faceLms.landmark))

            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = image.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                landmark.append((x, y))
                # print (id,(x,y))
        #         cv2.putText(image, str(id), (x, y),
        #                     cv2.FONT_HERSHEY_PLAIN, 0.1, (0, 255, 0), 1)
        # cv2.imwrite('out.jpg', image)
        # print(landmark)
        return landmark


def generate_478points_imgs(file):
    # 生成多张图片中的每张478人脸关键点
    # For static images:
    IMAGE_FILES = [file]
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # h, w = image.shape[0], image.shape[1]
            # image = cv2.resize(image, (int(w*8), int(h*8)), interpolation=cv2.INTER_LINEAR)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue

            annotated_image = image.copy()
            landmark = []
            for face_landmarks in results.multi_face_landmarks:
                # print('face_landmarks:', face_landmarks)
                faceLms = face_landmarks
                # print(len(faceLms.landmark))

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = image.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # print (id,(x,y))
                    cv2.putText(image, str(id), (x, y),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    landmark.append((x, y))
            return landmark
        cv2.imwrite(f'{file}_out.jpg', image)


def generator_oppo251(img_dir):
    # 生成251人脸关键点(图片目录)
    source_list = []
    base_dir = os.path.basename(img_dir)
    print(base_dir)
    for root, dirs, files in os.walk(img_dir, topdown=False):
        for file in files:
            source_list.append(os.path.join(root, file))
    # print(source_list)
    for s in source_list:
        target_dir = os.path.dirname(base_dir + s.split(img_dir)[1])
        # print(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        landmarks = generate_478points_imgs(s)
        ldmk251 = mediapipe478_251(landmarks)
        if not ldmk251:
            print("Failed to handle picture", s)
            continue
        file_basename = os.path.basename(s)
        # print(s)
        output = os.path.join(
            target_dir, os.path.splitext(file_basename)[0]+'_0.txt')
        # print(output)
        with open(output, 'w', encoding='utf-8') as w:
            w.write(f'{file_basename}\n')
            w.write('251\n')
            for id, p in enumerate(ldmk251):
                # print(id, p)
                w.write(f'{p[0]}\n')
                w.write(f'{p[1]}\n')
                w.write(f'1\n')


def generator_oppo251_points(img_file):
    # 生成251人脸关键点(单张图片)
    landmarks = generate_478points_imgs(img_file)
    print(len(landmarks))
    ldmk251 = mediapipe478_251(landmarks)

    # 在图片上绘点
    # image = cv2.imread(img_file)
    # h, w = image.shape[0], image.shape[1]
    # image = cv2.resize(image, (int(w*8), int(h*8)),
    #                    interpolation=cv2.INTER_LINEAR)
    # for id, p in enumerate(ldmk251):
    #     # print(id, x, y)
    #     cv2.putText(image, str(id), p, cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.3, (0, 255, 0), 1)
    #     # cv2.imshow('test', img)
    # cv2.imwrite(f'{img_file}_marked.jpg', image)
    return ldmk251


def test_oppo_251_txt(img, txt):
    image = cv2.imread(img)
    h, w = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(w*10), int(h*10)),
                       interpolation=cv2.INTER_LINEAR)
    with open(txt) as f:
        labels = [x.strip() for i, x in enumerate(
            f.readlines()[2:]) if (i+1) % 3 != 0]
        points = [(labels[i], labels[i+1])
                  for i, x in enumerate(labels) if i % 2 == 0]
        # print(points)
        for id, (x, y) in enumerate(points):
            x = round(float(x)*10)
            y = round(float(y)*10)
            # print(id, x, y)
            cv2.putText(image, str(id), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imwrite(f"{img}_marked.jpg", image)


def test_video(video_file):
    cap = cv2.VideoCapture(video_file)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    i = 0
    FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # CAP_PROP_POS_MSEC
    print(FRAMES)
    # cap.set(cv2.CAP_PROP_POS_FRAMES,160)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    success, image = cap.read()
    cv2.imwrite(f'{video_file}_test'+str(i)+'.jpg', image)
    # while i <= fps:
    #     success, image = cap.read(i)
    #     cv2.imwrite(f'{video_file}_test'+str(i)+'.jpg', image)
    #     i += 5
    cap.release()


def read_image_from_video(video_file, time):
    # print(time_fps)
    cap = cv2.VideoCapture(video_file)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(frames)
    cap.set(cv2.CAP_PROP_POS_MSEC, time)
    # cap.set(cv2.CAP_PROP_POS_FRAMES,time)
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        return
    # cv2.imwrite(f'{video_file}_test.jpg', image)
    cap.release()
    # print(image)
    return image


def clip_video2audio(video_file):
    # 从视频中获得音频文件
    my_audio_clip = AudioFileClip(video_file)
    name, suffix = os.path.splitext(os.path.basename(video_file))
    # print(stem, suffix)  # test   .py
    parent_path = os.path.dirname(video_file)
    # print(parent_path)
    my_audio_clip.write_audiofile(parent_path+"/video/"+name+".wav")
    print("*******音频分离成功*******")


def get_audio_word_timestamp():
    # 获取音频中每个词的时间戳，可根据实际目录改变目录
    # mfa align ./video ./mandarin-for-montreal-forced-aligner-pre-trained-model.txt ./mandarin.zip ./corpus
    # mfa align ./data/video12 ./MFA/mandarin-for-montreal-forced-aligner-pre-trained-model.txt ./MFA/mandarin.zip ./data/corpus
    # os.system("mfa align ./data/video ./MFA/mandarin-for-montreal-forced-aligner-pre-trained-model.txt ./MFA/mandarin.zip ./data/corpus")
    time_list1 = read_textgrid("./data/corpus/test2.TextGrid")
    time_list2 = read_textgrid("./data/corpus/gaoyan.TextGrid")
    print("*******已读取音频中每个词的时间戳*******")
    return time_list1, time_list2


def evaluate_video(video_file1, video_file2):
    # 评估视频唇形差异
    # clip_video2audio(video_file1) # 获取视频中的音频
    # clip_video2audio(video_file2)
    time_list1, time_list2 = get_audio_word_timestamp()  # 获取音频中每个词的时间戳
    length = min(len(time_list1), len(time_list2))
    # while cap.isOpened():
    sum_square = 0
    for i in range(length):
        print("正在进行评估......")
        image1 = read_image_from_video(video_file1, time_list1[i])
        image2 = read_image_from_video(video_file2, time_list2[i])
        # cv2.imwrite("test1_"+str(i)+".png", image1)
        landmarks1 = generate_478points_img(image1)
        landmarks2 = generate_478points_img(image2)
        ldmk1_251 = mediapipe478_251(landmarks1)
        ldmk2_251 = mediapipe478_251(landmarks2)
        dis1 = lips_opening_distance(ldmk1_251)
        dis2 = lips_opening_distance(ldmk2_251)
        sum_square += (dis1-dis2)*(dis1-dis2)
        # print (dis1)
        # print (dis2)
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        # cv2.imwrite("mediapipe1_"+str(i)+".png", image1)
        # cv2.imwrite("mediapipe2_"+str(i)+".png", image2)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    RMSE = math.sqrt(sum_square/length)
    name1, suffix = os.path.splitext(os.path.basename(video_file1))
    name2, suffix = os.path.splitext(os.path.basename(video_file2))
    print("评估结果：" + name1+" 与 " + name2 + " 的RMSE为：" + str(RMSE))
    return RMSE


def lips_opening_distance(ldmk251):
    # 通过251关键点计算唇部张开大小
    # lips_list = [191, 209, 200, 243]
    # for id in lips_list:
    #     cv2.putText(image, str(id), ldmk251[id], cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.3, (0, 255, 0), 1)
    #     print(id)
    #     print(ldmk251[id])
    x_center = (ldmk251[191][0]+ldmk251[209][0])/2
    y_center = (ldmk251[191][1]+ldmk251[209][1])/2
    # cv2.putText(image, str(0), (round(x_center), round(y_center)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.3, (0, 255, 0), 1)
    # print((x_center, y_center))
    dis_corners = distance(ldmk251[191], ldmk251[209])
    # print(dis_corners)
    ratio = 1/dis_corners
    dis1 = distance(ldmk251[216], (x_center, y_center))
    dis2 = distance(ldmk251[229], (x_center, y_center))
    dis_stretch = (dis1+dis2)*ratio
    # print(dis_stretch)
    # cv2.imwrite(f'{img_file}_marked.jpg', image)
    # print(image.shape)
    return dis_stretch


def read_textgrid(file):
    with open(file, 'r') as f:
        data = f.readlines()
        # print data #Use this to view how the code would look like after the program has opened the files
        txttext = ''
        time_list1 = []
        for lines in data[8:]:  # informations needed begin on the 9th lines
            # as there's \n at the end of every sentence.
            line = re.sub('\n', '', lines)
            line = re.sub('^ *', '', line)  # To remove any special characters
            if(line == "item [2]:"):
                break
            linepair = line.split(' = ')
            if len(linepair) == 2:
                if linepair[0] == 'xmin':
                    xmin = float(linepair[1])
                    # print(xmin)
                if linepair[0] == 'xmax':
                    xmax = float(linepair[1])
                if linepair[0] == 'text' and linepair[1] != '"" ':
                    xmid = (xmin+xmax)/2
                    time_list1.append(xmid)
                    # print(xmin)
                    # if linepair[1].strip().startswith('"') and linepair[1].strip().endswith('"'):
                    #     text = linepair[1].strip()[1:-1]
                    #     txttext += text + '\n'
        # print(time_list1)
        return time_list1


if __name__ == "__main__":
    # test_oppo_251_txt('image_160.jpg', 'image_160.txt') # GT picture draw
    # test_oppo_251_txt('image_001.jpg', 'image_001_0.txt')
    # generator_oppo251(r'E:\code\ftp\upload_files\FDD\ldmkvis')
    # test_oppo_251_txt('image_160.jpg', 'image_160_0.txt')
    # test_oppo_251_txt('image_160.jpg', 'image_160.txt')
    # generator_oppo251(r'E:\code\ftp\upload_files\FDD\21testyasuo')

    # static_img("./testdatas/1.png")
    # dis1 = lips_4points_distance_image("./testdatas/1.png")
    # dis2 = lips_4points_distance_image("./testdatas/2.png")
    # print(dis1-dis2)
    # evaluate_video('./testdatas/test1.mp4')
    # generator_oppo251_points('./testdatas/2.png')
    # test_video('./testdatas/test1.mp4')
    # image = read_image_from_video('./testdatas/test1.mp4', 5)
    # generate_478points_img(image)

    # get_audio_word_timestamp()
    # read_image_from_video('./testdatas/test1.mp4', 6*1000)
    # test_video('./testdatas/test1.mp4')
    # clip_video2audio('./testdatas/data/test1.mp4')
    # get_audio_word_timestamp()
    # get_audio_word_timestamp()
    # read_textgrid("./testdatas/data/corpus/test1.TextGrid")

    evaluate_video('./data/test2.mp4', './data/gaoyan.mp4')
    print("*******任务结束*******")
