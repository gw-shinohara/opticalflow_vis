import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LI_WIDTH=600 
LI_HEIGHT=480

class Opticalflow():
    def __init__(self, type="farneback"):
        self.type=type

    def process(self, previous_path, after_path):

        img1 = cv2.imread(previous_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)

        if self.type=="farneback":
            return self.farneback(img1, img2)
        elif self.type=="lucas_kanade":
            return self.lucas_kanade(img1, img2)
        else:
            return img1
    
    def farneback(self, img1, img2):
        # 画像を読み込み
        # オプティカルフローを計算
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # オプティカルフローの可視化
        hsv = np.zeros_like(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR))
        hsv[..., 1] = 255

        # フローから角度と速度を取得
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # HSVからBGRに変換して表示
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_img

    def lucas_kanade(self, img1, img2):
        features_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **features_params)

        # Lucas-Kanade法のパラメータ
        lk_params = dict(winSize=(80, 80), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # オプティカルフローを計算
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        # 有効な点をフィルタリング
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 結果の可視化
        output_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # 点と線を描画
            output_img = cv2.circle(output_img, (int(a), int(b)), 5, (0, 255, 0), -1)
            output_img = cv2.line(output_img, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2)
        return output_img


class ImgLabel(tk.Label):
    def __init__(self, root):
        super().__init__(root)
        self.path = None

    def set_path(self, path):
        self.__path = path

    def get_path(self):
        return self.__path

def load_image(path, label):
    img = Image.open(path)
    img = img.resize((LI_WIDTH, LI_HEIGHT), Image.FIXED)
    photo = ImageTk.PhotoImage(img)
    label.config(image=photo)
    label.image = photo

def open_image(img_label):
    file_path = filedialog.askopenfilename()
    img_label.set_path(file_path)
    if file_path:
        load_image(file_path, img_label)

def blend_image(pre_path, after_path):
    img1 = Image.open(pre_path)
    img1 = img1.resize((LI_WIDTH, LI_HEIGHT), Image.FIXED)
    img2 = Image.open(after_path)
    img2 = img2.resize((LI_WIDTH, LI_HEIGHT), Image.FIXED)
    blended_image = Image.blend(img1,img2, alpha=0.5)
    photo = ImageTk.PhotoImage(blended_image)
    return photo

def make_surface(root, pre_path):
    for widget in root.winfo_children():
        widget.destroy()
    image = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(LI_WIDTH, LI_HEIGHT),interpolation=cv2.INTER_NEAREST)

    # Get the x and y dimensions of the image
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    x, y = np.meshgrid(x, y)

    # Convert pixel values to floats to use as z-axis
    z = np.array(image, dtype=float)

    # Plotting the surface
    fig = plt.figure(figsize=(LI_WIDTH/100, LI_HEIGHT/100))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    surface = ax.plot_surface(x, y, z, cmap='inferno', edgecolor='none')

    # Optional: Customize the plot
    ax.set_title('3D Surface Plot of Grayscale Image')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Intensity')
    plt.gca().invert_yaxis()
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    ax.set_proj_type('ortho')

    def zoom(event):
        base_scale = 1.1
        if event.button == 'up':  # マウスホイールを上にスクロール
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # マウスホイールを下にスクロール
            scale_factor = base_scale
        else:
            return
        # マウスの位置に向かってズーム
        # 画面上のマウス位置（pixel単位）を取得
        x_mouse, y_mouse = event.x, event.y

        # マウス位置を軸の座標系（データ単位）に変換
        x_data =  float(ax.format_xdata(x_mouse))
        y_data =  float(ax.format_ydata(y_mouse))

        # 軸の現在の範囲を取得し、ズーム倍率に基づいて中心をシフト
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.set_xlim([x_data + (x - x_data) * scale_factor for x in xlim])
        ax.set_ylim([y_data + (y - y_data) * scale_factor for y in ylim])
        ax.set_zlim([z * scale_factor for z in zlim])
        fig.canvas.draw_idle()  # 図を再描画

    # イベントをバインド
    fig.canvas.mpl_connect('scroll_event', zoom)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

def compare_images(pre_path, after_path=None):
    if after_path == None:
        after_path = pre_path
    # Add comparison logic here
    print("Comparing images...")

    # farneback flow
    farneback = Opticalflow("farneback")
    cv_img = farneback.process(pre_path, after_path)
    img = Image.fromarray(cv_img)
    img = img.resize((LI_WIDTH, LI_HEIGHT), Image.FIXED)
    photo = ImageTk.PhotoImage(img)
    label3.config(image=photo)
    label3.image = photo

    # lucas_kanade flow
    lucas_kanade = Opticalflow("lucas_kanade")
    cv_img = lucas_kanade.process(pre_path, after_path)
    img = Image.fromarray(cv_img)
    img = img.resize((LI_WIDTH, LI_HEIGHT), Image.FIXED)
    photo = ImageTk.PhotoImage(img)
    label4.config(image=photo)
    label4.image = photo

    # blend image
    photo = blend_image(pre_path, after_path)
    label5.config(image=photo)
    label5.image = photo

    # surfase 3d
    make_surface(label6, pre_path)


if __name__ == '__main__':
    # Set up main window
    root = tk.Tk()
    root.title("Image Comparison")
    root.geometry("1280x960")

    # Left Column - Original images
    label1 = ImgLabel(root)
    label1.grid(row=0, column=0)
    btn1 = tk.Button(root, text="Open Image 1", command=lambda: open_image(label1))
    btn1.grid(row=0, column=0, sticky="s")

    label2 = ImgLabel(root)
    label2.grid(row=1, column=0)
    btn2 = tk.Button(root, text="Open Image 2", command=lambda: open_image(label2))
    btn2.grid(row=1, column=0, sticky="s")

    # Right Column - Processed images
    label3 = tk.Label(root)
    label3.grid(row=0, column=1)

    label4 = tk.Label(root)
    label4.grid(row=1, column=1)

    label5 = tk.Label(root)
    label5.grid(row=0, column=2)

    # Compare Button
    compare_btn = tk.Button(root, text="Compare", command=lambda: compare_images(label1.get_path(), label2.get_path()))
    compare_btn.grid(row=2, column=0, columnspan=2)

    label6 = tk.Label(root)
    label6.grid(row=1, column=2)

    root.mainloop()
