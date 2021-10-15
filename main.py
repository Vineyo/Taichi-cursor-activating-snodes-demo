import taichi as ti

ti.init(arch=ti.cuda)

@ti.data_oriented
class sparse_canvas:
    def __init__(self):
        
        self.n = 512
        self.x = ti.field(dtype=ti.i32)
        self.y = ti.field(dtype=ti.i32)
        self.cursor=ti.field(dtype=float,shape=2)

        self.res = self.n + self.n // 4 + self.n // 16 + self.n // 64
        self.img = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self.tip=ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self.block1 = ti.root.pointer(ti.ij, self.n // 64)
        self.block2 = self.block1.pointer(ti.ij, 4)
        self.block3 = self.block2.dense(ti.ij, 4)
        self.block3.dense(ti.ij, 4).place(self.x)
        self.block3.dense(ti.ij, 4).place(self.y)


    @ti.kernel
    def activate(self,t: ti.f32):
        for i, j in ti.ndrange(self.n, self.n):
            p = ti.Vector([i, j]) / self.n
            p = ti.Matrix.rotation2d(ti.sin(t)) @ (p - 0.5) + 0.5

            if ti.taichi_logo(p) == 0:
                self.x[i, j] = 1


    @ti.func
    def scatter(self,i):
        return i + i // 4 + i // 16 + i // 64 + 2

    @ti.kernel
    def draw(self):
        center = ti.Vector([self.cursor[0], self.cursor[1]])
        for i, j in ti.ndrange(self.n, self.n):
            dis = (ti.Vector([i, j])/self.n-center).norm()
            if dis < 0.03:
                self.tip[i,j]+=0.3

    @ti.kernel
    def tip_to_x(self):
        for P in ti.grouped(self.tip):
            if self.tip[P]>0:
                self.x[P]=self.tip[P]
                self.tip[P]=0

    @ti.kernel
    def paint(self):
        for i, j in ti.ndrange(self.n, self.n):
            t = self.y[i, j]
            block1_index = ti.rescale_index(self.y, self.block1, [i, j])
            block2_index = ti.rescale_index(self.y, self.block2, [i, j])
            if ti.is_active(self.block1, block1_index):
                t += 1
            if ti.is_active(self.block2, block2_index):
                t += 1
            self.img[self.scatter(i), self.scatter(j)] = 1 - t / 4

canvas=sparse_canvas()
canvas.img.fill(0.05)

gui = ti.GUI('Sparse Grids', (canvas.res, canvas.res))

for i in range(100000):
    gui.get_event()
    if(gui.is_pressed(ti.GUI.LMB)):
        canvas.cursor[1]=gui.get_cursor_pos()[1]
        canvas.cursor[0]=gui.get_cursor_pos()[0]
        canvas.draw()
        
    canvas.tip_to_x()
    canvas.paint()
    gui.set_image(canvas.img)
    gui.show()
