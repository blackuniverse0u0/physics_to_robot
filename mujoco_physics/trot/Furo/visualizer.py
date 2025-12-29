import matplotlib.pyplot as plt
import matplotlib
# macOS에서 별도 프로세스로 실행 시 TkAgg가 가장 안정적입니다.
matplotlib.use('TkAgg') 
from collections import deque
import time

class PlotProcess:
    def __init__(self, queue, buffer_len=200):
        self.queue = queue
        self.buffer_len = buffer_len
        self.time_buffer = deque(maxlen=buffer_len)
        self.pos_data = [deque(maxlen=buffer_len) for _ in range(3)]
        self.vel_data = [deque(maxlen=buffer_len) for _ in range(3)]
        self.trq_data = [deque(maxlen=buffer_len) for _ in range(3)]
        
        self.init_plot()

    def init_plot(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        self.fig.canvas.manager.set_window_title("Real-Time Joint Data")
        
        self.lines_pos = []
        self.lines_vel = []
        self.lines_trq = []
        
        labels = ['Hip Roll', 'Hip Pitch', 'Knee']
        colors = ['r', 'g', 'b']
        
        titles = ['Position (rad)', 'Velocity (rad/s)', 'Torque (Nm)']
        
        for i in range(3): # 3 Subplots
            self.axs[i].set_ylabel(titles[i])
            self.axs[i].grid(True)
            lines = []
            for j in range(3): # 3 Joints per subplot
                line, = self.axs[i].plot([], [], label=labels[j], color=colors[j])
                lines.append(line)
            if i == 0: self.axs[i].legend(loc='upper right')
            
            if i == 0: self.lines_pos = lines
            elif i == 1: self.lines_vel = lines
            elif i == 2: self.lines_trq = lines
            
        plt.tight_layout()

    def run(self):
        last_draw_time = time.time()
        while True:
            # 큐에서 데이터가 있으면 모두 꺼냄 (최신 상태 유지)
            while not self.queue.empty():
                try:
                    data = self.queue.get_nowait()
                    if data is None: # 종료 신호
                        plt.close('all')
                        return
                    
                    t, pos, vel, trq = data
                    self.time_buffer.append(t)
                    for i in range(3):
                        self.pos_data[i].append(pos[i])
                        self.vel_data[i].append(vel[i])
                        self.trq_data[i].append(trq[i])
                except:
                    break
            
            # 그래프 갱신 (초당 10프레임 정도로 제한하여 부하 감소)
            if time.time() - last_draw_time > 0.1:
                if self.time_buffer:
                    t_list = list(self.time_buffer)
                    for i in range(3):
                        self.lines_pos[i].set_data(t_list, list(self.pos_data[i]))
                        self.lines_vel[i].set_data(t_list, list(self.vel_data[i]))
                        self.lines_trq[i].set_data(t_list, list(self.trq_data[i]))
                    
                    for ax in self.axs:
                        ax.relim()
                        ax.autoscale_view(True, True, True)
                        ax.set_xlim(max(0, t_list[-1] - 5), t_list[-1] + 0.1)
                        
                    self.fig.canvas.flush_events()
                last_draw_time = time.time()

# 이 함수를 메인 프로세스에서 호출하여 실행
def launch_plotter(queue):
    plotter = PlotProcess(queue)
    plotter.run()
