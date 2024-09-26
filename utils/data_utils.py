import numpy as np
import json
import cv2



def get_points_from_CVAT_xml(imageName, points_list, xml_imageEles):
    labels = []
    x_pos = []
    y_pos = []

    for imageEle in xml_imageEles:
        if imageName == imageEle.attrib['name']:
            for i, pointName in enumerate(points_list):
                for pointEle in imageEle.iter('points'):
                    if pointName == pointEle.attrib['label']:
                        labels.append(pointEle.attrib['label'])
                        x_pos.append(float(pointEle.attrib['points'].split(',')[0]))
                        y_pos.append(float(pointEle.attrib['points'].split(',')[1]))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    # print(imageName, labels, x_pos, y_pos)

    return labels, x_pos, y_pos

def get_tagName_from_CVAT_xml(imageEle):
    for tagEle in imageEle.iter('tag'):
        return tagEle.attrib['label']


def get_points_from_txt(point_num, path):
    flag = 0
    x_pos = []
    y_pos = []
    points_name = []
    with open(path) as note:
        for line in note:
            if flag >= point_num:
                break
            else:
                flag += 1
                x, y = [float(i) for i in line.split(',')]
                x_pos.append(x)
                y_pos.append(y)
                points_name.append(str(flag))
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
    return points_name, x_pos, y_pos


def get_points_from_json(point_list, json_path):
    f = open(json_path, 'r')
    data = json.loads(f.read())

    labels = []
    x_pos = []
    y_pos = []

    for i in range(0,len(point_list)):
        points_value = data["markerPoints"][point_list[i]]
        labels.append(point_list[i])
        x_pos.append(points_value.get("x"))
        y_pos.append(points_value.get("y"))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    return labels, x_pos, y_pos

def get_predict_point_from_heatmap(heatmaps, scal_ratio_w, scal_ratio_h, points_num):

    pred = np.zeros((points_num, 2))
    for i in range(points_num):
        heatmap = heatmaps[i]
        pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        pred[i][1] = pre_y[0] * scal_ratio_h
        pred[i][0] = pre_x[0] * scal_ratio_w
    return pred


def visualization(image_path, points_x, points_y, points_num, points_name):
    image = cv2.imread(image_path, 1)

    for j in range(points_num):
        x, y = int(points_x[j]), int(points_y[j])
        point_name = points_name[j]
        output_image = cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=2)
        output_image = cv2.putText(output_image, str(point_name), (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (0, 0, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized_Window", output_image)
    cv2.waitKey(0)




def cross_distance_calculation(p1, p2, p3, ratio=False):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    d_pq_p2 = np.linalg.norm(p2 - p1)
    output = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    if ratio:
        output = output / d_pq_p2

    return output

def euclid_distance_calculation(p1, p2):

    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p2 - p1)



def angle_calculation(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    p3_p1 = p3 - p1
    p3_p2 = p3 - p2

    cosine_angle = np.dot(p3_p1, p3_p2) / (np.linalg.norm(p3_p1) * np.linalg.norm(p3_p2))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)



def CVM_pattern_calculation(json_file_path):
    f = open(json_file_path, 'r')
    data = json.loads(f.read())
    C2a = data["markerPoints"]['C2a']
    C2p = data["markerPoints"]['C2p']
    C2m = data["markerPoints"]['C2m']
    print(C2a)
    print(type(C2a))

def calculate_CVM_metrics_from_json(input_json_path, CVM_list):
    points_name, gt_x, gt_y = get_points_from_json(CVM_list, input_json_path)

    C2a = [(gt_x[0]), (gt_y[0])]
    C2m = [(gt_x[1]), (gt_y[1])]
    C2p = [(gt_x[2]), (gt_y[2])]
    C3ua = [(gt_x[3]), (gt_y[3])]
    C3up = [(gt_x[4]), (gt_y[4])]
    C3la = [(gt_x[5]), (gt_y[5])]
    C3m = [(gt_x[6]), (gt_y[6])]
    C3lp = [(gt_x[7]), (gt_y[7])]
    C4ua = [(gt_x[8]), (gt_y[8])]
    C4up = [(gt_x[9]), (gt_y[9])]
    C4la = [(gt_x[10]), (gt_y[10])]
    C4m = [(gt_x[11]), (gt_y[11])]
    C4lp = [(gt_x[12]), (gt_y[12])]
    C2s = [(gt_x[13]), (gt_y[13])]

    ###### Calculate CVM features #####

    C2Angle = angle_calculation(C2a, C2p, C2m)
    C3Angle = angle_calculation(C3la, C3lp, C3m)
    C4Angle = angle_calculation(C4la, C4lp, C4m)

    C2Conc = cross_distance_calculation(C2a, C2p, C2m, ratio=False)
    C3Conc = cross_distance_calculation(C3la, C3lp, C3m, ratio=False)
    C4Conc = cross_distance_calculation(C4la, C4lp, C4m, ratio=False)

    C3PAR = euclid_distance_calculation(C3up, C3lp) / euclid_distance_calculation(C3ua, C3la)
    C3BAR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3ua, C3la)
    C3BPR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3up, C3lp)

    C4PAR = euclid_distance_calculation(C4up, C4lp) / euclid_distance_calculation(C4ua, C4la)
    C4BAR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4ua, C4la)
    C4BPR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4up, C4lp)

    return C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle




def calculate_CVM_metrics_from_inference_result(pred, CVM_list):

    C2a = [(pred[0][0]), (pred[0][1])]
    C2m = [(pred[1][0]), (pred[1][1])]
    C2p = [(pred[2][0]), (pred[2][1])]
    C3ua = [(pred[3][0]), (pred[3][1])]
    C3up = [(pred[4][0]), (pred[4][1])]
    C3la = [(pred[5][0]), (pred[5][1])]
    C3m = [(pred[6][0]), (pred[6][1])]
    C3lp = [(pred[7][0]), (pred[7][1])]
    C4ua = [(pred[8][0]), (pred[8][1])]
    C4up = [(pred[9][0]), (pred[9][1])]
    C4la = [(pred[10][0]), (pred[10][1])]
    C4m = [(pred[11][0]), (pred[11][1])]
    C4lp = [(pred[12][0]), (pred[12][1])]

    ###### Calculate CVM features #####

    C2Angle = angle_calculation(C2a, C2p, C2m)
    C3Angle = angle_calculation(C3la, C3lp, C3m)
    C4Angle = angle_calculation(C4la, C4lp, C4m)

    C2Conc = cross_distance_calculation(C2a, C2p, C2m, ratio=False)
    C3Conc = cross_distance_calculation(C3la, C3lp, C3m, ratio=False)
    C4Conc = cross_distance_calculation(C4la, C4lp, C4m, ratio=False)

    C3PAR = euclid_distance_calculation(C3up, C3lp) / euclid_distance_calculation(C3ua, C3la)
    C3BAR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3ua, C3la)
    C3BPR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3up, C3lp)

    C4PAR = euclid_distance_calculation(C4up, C4lp) / euclid_distance_calculation(C4ua, C4la)
    C4BAR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4ua, C4la)
    C4BPR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4up, C4lp)

    return C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle



# test_json = 'D:/Self-projects/2D_Cephalometry/data/steinerAnno/train/points/1589963708196-573300089.json'
# get_points_from_json(17, test_json)