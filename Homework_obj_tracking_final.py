#! /usr/bin/python3.7
from importRosbag.importRosbag import importRosbag
from pathlib import Path
import sys
import subprocess
import os
import pickle
from dataclasses import dataclass
import copy

try:
    import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from screeninfo import get_monitors

except:
    subprocess.check_call([sys.executable,'-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable,'-m', 'pip', 'install', '-r', 'requirements.txt'])

    import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from screeninfo import get_monitors

@dataclass
class Object:
    x : float
    y : float
    vx : float
    vy : float
    ax : float
    ay : float
    heading : float
    life : int
    cov : np.array([])

class Object_array:
    # info : vx,vy, head
    def __init__(self):
        self.list = []

    def update(self, data, info = None):
        self.list = []

        for i in range(len(data)):
            self.list.append(Object(x=data[i,0], y=data[i,1], vx=data[i,2], vy=data[i,3], ax=0, ay=0, heading=data[i,4], life=1, cov=np.array([])))

    def num(self):
        return len(self.list)


def get_info_in_frame(idx):

    points_by_time_low =  topics["/lidar_low_level"]["point"][topics["/lidar_low_level"]["ts"]==time_sequence_low[i],:]

    idx = np.argmin(abs(time_sequence_high_xy - time_sequence_low[i]))
    points_by_time_high_xy = topics["/lidar_high_level_xy"]["point"][topics["/lidar_high_level_xy"]["ts"]==time_sequence_high_xy[idx],:]

    idx = np.argmin(abs(time_sequence_high_vxy_yaw - time_sequence_low[i]))
    points_by_time_high_vxy_yaw = topics["/lidar_high_level_vxy_yaw"]["point"][topics["/lidar_high_level_vxy_yaw"]["ts"]==time_sequence_high_xy[idx],:]

    idx = np.argmin(abs(topics["/image_reference"]["ts"] - time_sequence_low[i]))
    image = topics["/image_reference"]["frames"][idx]

    shape = np.shape(points_by_time_high_xy)
    
    data_high = np.array([])
    data_high.resize(shape[0], shape[1]+2)
        
    data_high[:,0] = points_by_time_high_xy[:,0] ## x
    data_high[:,1] = points_by_time_high_xy[:,1] ## y
    data_high[:,2] = points_by_time_high_vxy_yaw[:,0] ## vx
    data_high[:,3] = points_by_time_high_vxy_yaw[:,1] ## vy
    data_high[:,4] = points_by_time_high_vxy_yaw[:,2] ## heading

    return data_high, points_by_time_low, image


def Track_initiation(current_obj, previous_obj):
    initiated_track = Object_array()
    ## time k(현재)와 time k-1 의 센서 데이터를 비교한다. --> measurement to measurement association
    ## Association 결과는 Binary matrix 등으로 표현이 가능
    ## Association 방법은 Nearest Neighbor(NN) 을 사용 - 다른 방법을 사용해도 무방
    ## Association 된 Object 에 대해서는 Scoring 을 하여 점수를 누적한다
    ## 일정 횟수 이상 Association 된 물체는 따로 Tracking 하기 위해서 Initiating 한다. --> 새로운 Track을 만든다
    

    ## SENSOR DATA REPRESENTATION
    ## current_obj.list[0].x        -->  현재 센서 데이터 중 첫 번째 오브젝트의 x
    ## current_obj.list[0].y        -->  현재 센서 데이터 중 첫 번째 오브젝트의 y
    ## ....
    ## current_obj.list[0].heading  -->  현재 센서 데이터 중 첫 번째 오브젝트의 heading
    # print(current_obj.list[0])

    ## current_obj.num()     -->  감지된 High-level 데이터 개수(=감지된 오브젝트 데이터 개수)
    # print(current_obj.num(), ' objects in the current frame.')


    ## 예를 들어, current_obj 의 3번째 데이터를 Track 으로 Initiating 시키고 싶은 경우 
    # initiated_track.list.append(Object(x=current_obj[3].x,
    #                                    y=current_obj[3].y,
    #                                    vx=current_obj[3].vx,
    #                                    vy=current_obj[3].vy,
    #                                    ax=current_obj[3].ax,
    #                                    ay=current_obj[3].ay,
    #                                    heading=current_obj[3].heading,
    #                                    life=life_init,
    #                                    cov=np.eye(state_num)))
    # 거리행렬 생성
    dist=np.zeros((current_obj.num(),previous_obj.num()))
    Score=np.zeros((current_obj.num()))
    dist_min=1
    for i in range((current_obj.num())):
        for j in range((previous_obj.num())):
            dist[i,j]=((current_obj.list[i].x-previous_obj.list[j].x)**2+(current_obj.list[i].y-previous_obj.list[j].y)**2)**(1/2)
        
    for i in range(current_obj.num()):
        """Do something here"""
        for j in range(previous_obj.num()):
            """Do something here"""
            if (dist[i,j] < dist_min) :
                Score[i]=Score[i]+1
            else:
                Score[i]=Score[i]

    for j in range(previous_obj.num()):
        for i in range(current_obj.num()):
            if (Score[i]>=1):
                initiated_track.list.append(Object(x=current_obj.list[i].x,
                                       y=current_obj.list[i].y,
                                       vx=current_obj.list[i].vx,
                                       vy=current_obj.list[i].vy,
                                       ax=current_obj.list[i].ax,
                                       ay=current_obj.list[i].ay,
                                       heading=current_obj.list[i].heading,
                                       life=life_init,
                                       cov=np.eye(state_num)))
    return initiated_track


def Track_merge(initiated_track, your_track):
    ## 기존에 관리되던 Track 과 방금 생긴 Initiated_track을 병합

    for i in range(initiated_track.num()):
        your_track.list.append(initiated_track.list[i])

    return your_track


def Track_to_measurement_association(current_obj, your_track):
    ## 어떤 Track 과 어떤 Measurement 가 연관되어 있는지 알기 위해 track to measurement association 을 수행
    ## Association 결과는 Binary matrix 등으로 표현이 가능
    ## Association 방법은 Nearest Neighbor(NN) 을 사용(다른 방법도 가능)

    ## Association 정보 변수 초기화
    asso_info = []
    # print(your_track.num())
    for i in range(your_track.num()):
        asso_info.append(0)
    
    ## Association 수행(여러분이 코딩하시면 됩니다)
    for i in range(your_track.num()):
        dist_min = 1
        asso_possible=0
        for j in range(current_obj.num()):
            dist_current = ((your_track.list[i].x-current_obj.list[j].x)**2+(your_track.list[i].y-current_obj.list[j].y)**2)**(1/2)
            if (dist_current<dist_min):
                asso_idx = j
                asso_possible=1    
        if (asso_possible==1):      
            """if j'th measurement is corresponding to the i'th track"""
            asso_info[i] = current_obj.list[asso_idx]
        else:
            """if not"""
            asso_info[i] = 0

    return asso_info


def Track_state_estimation(asso_info, your_track):
    ## Track_to_measurent_association 결과 Track 에 Association 된 Measurement 가 있는 것으로 판단되는 경우 해당 Track 에 대해
    ## State estimation 을 수행해 Measurement 의 noise 를 저감하고 실제 차량의 움직임을 모사한다.
    ## Esimation 은 Kalman filter 를 사용(칼만필터 외에 다른 State observer 사용 가능)
    ## Kalman filter 예제는 아래
    
    for i in range(your_track.num()):

        x = np.array([])

        if choose_motion_model == 'CV':
            x = np.array([your_track.list[i].x,
                          your_track.list[i].y,
                          your_track.list[i].vx,
                          your_track.list[i].vy])

        elif choose_motion_model == 'CA':
            x = np.array([your_track.list[i].x,
                          your_track.list[i].y,
                          your_track.list[i].vx,
                          your_track.list[i].vy,
                          your_track.list[i].ax,
                          your_track.list[i].ay])

        P = your_track.list[i].cov

        ## Prediction step of Kalman filter
        xp = F@x
        Pp = F@P@F.transpose() + Q

        if asso_info[i] != 0: ## Association 된 경우에만 Correction step 을 진행한다.
            ## Calculating Kalman gain
            K = np.array([])
            K.resize(np.shape(F))
            K = Pp@H.transpose() @ np.linalg.inv(H@Pp@H.transpose() + R)
            
            z = np.array([])
            z.resize(np.shape(H)[0],1)

            ## You should find appropriate measurement which is associated to the i'th track
            z = np.array([asso_info[i].x,
                          asso_info[i].y,
                          asso_info[i].vx,
                          asso_info[i].vy])

            ## Correction step of Kalman filter
            x = xp + K@(z - H@xp)
            P = Pp - K@H@Pp

        else: ## Association 이 안된 Track은 Prediction step 만 수행
            x = xp
            P = Pp

        if choose_motion_model == 'CV':
            your_track.list[i].x   = x[0]
            your_track.list[i].y   = x[1]
            your_track.list[i].vx  = x[2]
            your_track.list[i].vy  = x[3]
            your_track.list[i].cov = P

        elif choose_motion_model == 'CA':
            your_track.list[i].x   = x[0]
            your_track.list[i].y   = x[1]
            your_track.list[i].vx  = x[2]
            your_track.list[i].vy  = x[3]
            your_track.list[i].ax  = x[4]
            your_track.list[i].ay  = x[5]
            your_track.list[i].cov = P

    return your_track


def Track_management(your_track):
    # 동일한 물체에 대해 Tracking 을 중복으로 수행할 필요가 없으므로, 기존에 관리되고 있는 Track 과 새로 생긴 Track 간의 위치를 비교해 너무 가까운 Track 중 하나만 남도록 제거
    ## Track_to_measurent_association 결과 Measurement 와 Association 된 Track 들은 life 를 증가, Association 이 되지 않은 Track 들은 life 감소
    ## life 는 상한선이 있으며, 지속적으로 Association 되지 않은 Track 은 life 가 0 이 될 것이고 해당 Track 은 더이상 Tracking 할 필요가 없으므로 제거
    
    # 자신의 Track을 제외한 가장 가까운 Track을 제거
    dist_min=1
    for i in reversed(range(your_track.num())):
        del_possible=0
        for j in range(i):
            track_dist = ((your_track.list[i].x-your_track.list[j].x)**2+(your_track.list[i].y-your_track.list[j].y)**2)**(1/2)
            if (track_dist<dist_min):
                del_possible=1
        if (del_possible==1):
            del your_track.list[i]
             
    
    # Track Life 관리
    for i in range(your_track.num()):
        if asso_info[i]==0:
            your_track.list[i].life -= 1
        else:
            your_track.list[i].life += 1

    
    # Track 상한선 및 Life 0되면 제거
    for i in reversed(range(your_track.num())):
        print(i)
        if your_track.list[i].life >= life_max:
            your_track.list[i].life = life_max
        
        if your_track.list[i].life == 0:
            del your_track.list[i]
            
    return your_track


def plot(high_lidar, low_lidar, your_track, image, dt):

    ## 데이터 변환
    high_lidar_arr = np.array([])
    high_lidar_arr.resize(high_lidar.num(), 2)

    for i in range(high_lidar.num()):
        high_lidar_arr[i,:] = [high_lidar.list[i].x, high_lidar.list[i].y]

    your_track_arr = np.array([])
    your_track_arr.resize(your_track.num(), 2)

    for i in range(your_track.num()):
        your_track_arr[i,:] = [your_track.list[i].x, your_track.list[i].y]
    
    ## 센서 데이터 및 추정 결과 도시
    plt.figure(0)
    plt.get_current_fig_manager().resize(monitor_width*0.25, monitor_height*0.9)
    plt.get_current_fig_manager().window.wm_geometry("+0+0")
    plt.clf()
    plt.scatter(low_lidar[:,1],      low_lidar[:,0],      s=0.1,  color='blue',   alpha=0.4,   label='Low-level data')
    plt.scatter(high_lidar_arr[:,1], high_lidar_arr[:,0], s=10,   color='red',    alpha=1,     label='High-level data')
    plt.scatter(your_track_arr[:,1], your_track_arr[:,0], s=100,  color='green',  alpha=0.5,   label='Your track data', marker='^')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15))    
    plt.grid(1, alpha=0.2)
    plt.xlim(-30, 30)
    plt.ylim(-80, 80)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.title('Highway Multi-Object Tracking')
    plt.xlabel('X axis [m]')
    plt.ylabel('Y axis [m]')
    

    ## 자차 모양 - 직사각형
    shp = patches.Rectangle((-0.75,-2), 1.5,4, color='k')
    plt.gca().add_patch(shp)

    ## x축 - 화살표
    shp = patches.Arrow(0,0,0,15.5,width=1,edgecolor='k',facecolor='k',alpha=0.3)
    plt.gca().add_patch(shp)
    plt.text(1,17.5,'x',alpha=0.3)

    ## y축 - 화살표
    shp = patches.Arrow(0,0,15.5,0,width=1,edgecolor='k',facecolor='k',alpha=0.3)
    plt.gca().add_patch(shp)
    plt.text(18.5,-1,'y',alpha=0.3)
    
    ## 참고용 이미지 표시
    plt.figure(1)
    plt.clf()
    plt.title('Image for reference')
    plt.imshow(image)
    plt.get_current_fig_manager().window.wm_geometry("+850+0")

    plt.pause(dt)


if __name__=="__main__":
    ####################################
    ## 여러분이 선택할 것 
    ## Motion model: CV or CA
    choose_motion_model = 'CA'
    ## Choose Scene number: 0, 1, 2
    choose_scene_num = 0
    ####################################
    ####################################

    ## 데이터 타입 변환 및 기초 작업 (원하는대로 수정해도 되지만 크게 건드릴 부분은 없을 것 같습니다.)
    previous_obj = Object_array()
    current_obj  = Object_array()
    your_track   = Object_array()

    life_init = 5
    life_max = 10
    
    dt = 0.1
    seq = 0

    monitor_width  = get_monitors()[0].width
    monitor_height = get_monitors()[0].height

    plot_width  = monitor_width*0.25
    plot_height = monitor_height*0.9

    dir_path = os.getcwd()
    total_scene_num = 3

    for i in range(total_scene_num):
        bag_name    = 'scene_' + str(i) + '.bag'
        pickle_name = 'scene_' + str(i) + '.pkl'

        bag_path    = dir_path + '\\' + bag_name
        pickle_path = dir_path + '\\' + pickle_name

        if os.path.exists(pickle_path):
            print(pickle_name, "already exists")

        else:
            importTopics = ['/lidar_high_level_xy', '/lidar_high_level_vxy_yaw', '/lidar_low_level', '/chassis_ego', '/image_reference']
            topics = importRosbag(filePathOrName = bag_path , importTopics = importTopics)

            with open(pickle_name, 'wb') as f:
                pickle.dump(topics, f)

    pickle_path = dir_path + '\scene_' + str(choose_scene_num) + '.pkl'

    sensor_data = open(pickle_path, "rb")
    topics = pickle.load(sensor_data)

    time_sequence_low          = np.unique(topics["/lidar_low_level"]["ts"])
    time_sequence_high_xy      = np.unique(topics["/lidar_high_level_xy"]["ts"])
    time_sequence_high_vxy_yaw = np.unique(topics["/lidar_high_level_vxy_yaw"]["ts"])

    F = np.array([])
    H = np.array([])
    Q = np.array([])
    R = np.array([])
    # Kalman Gain 크면 측정값 신뢰도 up(Q커지면 Kalman Gain 커짐) => R값을 크게하여 Model에 대한 신뢰도 up
    if choose_motion_model == 'CV':
        ## state vector x = [x y vx vy]'
        ## Discrete-time state transition matrix
        F = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])

        ## Observation matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        ## Model uncertainty (여러분이 파라미터 튜닝을 하시면 됩니다)
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        ## Measurement uncertainty (여러분이 파라미터 튜닝을 하시면 됩니다)
        R = np.array([[1.5, 0, 0, 0],
                      [0, 1.5, 0, 0],
                      [0, 0, 1.5, 0],
                      [0, 0, 0, 1.5]])

    elif choose_motion_model == 'CA':
        ## state vector x = [x y vx vy ax ay]'
        ## Discrete-time state transition matrix(수정)
        F = np.array([[1, 0, dt,  0, 0.5*dt*dt, 0],
                      [0, 1,  0, dt, 0, 0.5*dt*dt],
                      [0, 0,  1,  0, 0,         0],
                      [0, 0,  0,  1, 0,         0],
                      [0, 0,  0,  0, 1,         0],
                      [0, 0,  0,  0, 0,         1]])

        ## Observation matrix
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])


        ## Model uncertainty (여러분이 파라미터 튜닝을 하시면 됩니다)
        Q = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        ## Measurement uncertainty (여러분이 파라미터 튜닝을 하시면 됩니다)
        R = np.array([[1.5, 0, 0, 0],
                      [0, 1.5, 0, 0],
                      [0, 0, 1.5, 0],
                      [0, 0, 0, 1.5]])
    
    state_num = np.shape(F)[0]
    P_init = np.eye(state_num)

    ## 아래 for 문 내부에 여러분 코드를 작성하면 됩니다.
    for i in range(len(time_sequence_low)):

        ## 시간 차이(Time differnce) 계산(수정 불필요)
        try:
            dt = time_sequence_low[i+1] - time_sequence_low[i]
        except:
            dt = 0.1
        
        ## Sequence에서 frame별 값 추출(수정 불필요)
        data_high, data_low, image = get_info_in_frame(i)
        ## Frame Update(수정 불필요)
        current_obj.update(data_high)


        #############################################################################################
        ## 여러분이 작성해야 할 함수 ################################################################## 
        initiated_track = Track_initiation(current_obj, previous_obj)
        your_track      = Track_merge(initiated_track, your_track)
        asso_info       = Track_to_measurement_association(current_obj, your_track)
        your_track      = Track_state_estimation(asso_info, your_track)
        your_track      = Track_management(your_track)
        #############################################################################################
        #############################################################################################

        ## 움직이는 plot 그리기(원하는대로 수정 가능)
        plot(current_obj, data_low, your_track, image, dt)

        ## Previous Frame Update(다음 Time sequence에서 k-1 step 정보로 활용 가능하도록 데이터 저장)
        previous_obj = copy.deepcopy(current_obj)
        seq += 1

    plt.show()
