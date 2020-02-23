class LBP:
    def __init__(self, window_size=(128, 68), cell_size=(16,16), delta=(8,8), num_nn=8, radius=1):
        self.window_size = window_size
        self.cell_size = cell_size
        self.delta = delta
        self.num_nn = num_nn
        self.radius = radius

    def compute(self, img, method="basic"):
        img_x_size, img_y_size = img.shape

