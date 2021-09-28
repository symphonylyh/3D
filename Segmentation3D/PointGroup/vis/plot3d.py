'''
Utility for visualizing 3D point cloud. Based on Open3D.
Main functinalities:
    - Use GUI scene to plot multiple point clouds in different viewports
    - Sync the camera view across different plots
    - Press 'S' to save snapshot of each scene
Good examples from Open3D developers:
    https://github.com/intel-isl/Open3D/tree/master/examples/python/gui
Usage:
    see /plot3d_examples/
'''
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os, sys
import numpy as np
import colorsys
import random

'''
GUI level control. A GUI is made of many figure windows (each of them is a Plot3DFigure object).
'''
class Plot3DApp:
    def __init__(self):
        # gui: high-level handle of the GUI application consisting of many figures (one figure is one window)
        self.gui = gui.Application.instance
        self.gui.initialize()

        # figures: {"figure_name" : figure} dictionary storing different figures by their name
        self.figures = {}

        # running: True if Plot3DApp.plot() has been called before
        self.running = False

    def create_figure(self, figure_name, **kwargs):
        '''
        Create figure window.
        Argument list see Plot3DFigure class.
        '''
        if figure_name == '':
            figure_name = "Figure " + str(len(self.figures) + 1) 

        if figure_name in self.figures.keys():
            self.close()
            sys.exit(f"figure name '{figure_name}' already exists")

        figure = Plot3DFigure(self, figure_name, **kwargs)
        self.figures[figure_name] = figure 

        return figure
    
    def get_figure(self, name):
        '''
        Get figure handle by name.
        '''
        figure = self.figures.get(name)
        if figure is None:
            self.close()
            sys.exit(f"figure name '{name}' doesn't exist")
        else:
            return figure  

    def plot(self):
        '''
        Start plotting figures.
        '''
        self.running = True
        self.gui.run()

    def close(self):
        '''
        Close all figures/windows and quit application.
        '''   
        # close all windows
        for figure in self.figures.values():
            figure.close()

        # if run() is not called, call it first before quit(), otherwise there is a deadlock
        if not self.running:
            self.gui.run()

        # quit() is not even necessary. I found that closing all windows and then call run() will just exit
        self.gui.quit()

'''
Subfigure level control. Each subfigure is a SceneWidget object, but it also stores back pointer to its parent figure as well as its row/col location in the parent figure.
'''
class Plot3DSubFigure():
    def __init__(self, parent, row, col, handle):
        self.parent = parent 
        self.row = row
        self.col = col 
        self.handle = handle

'''
Figure level control. A figure is made of many subplots (each of them is a SceneWidget object).
'''
class Plot3DFigure:
    def __init__(self, app, figure_name, viewports_dim=(1,1), width=1920, height=1080, sync_camera=True, plot_boundary=True, show_axes=True, show_subtitles=True, background_color=(0,0,0,1), snapshot_path='./'):
        self.app = app
        self.name = figure_name
        self.viewports_dim = viewports_dim
        self.rows, self.cols = self.viewports_dim
        self.window = self.app.gui.create_window(figure_name, width, height)
        self.sync_camera = sync_camera # True: sync cameras in subplots; False: individual cameras
        self.plot_boundary = plot_boundary # True: plot black boundary around subplots; False: no boundary
        self.show_axes = show_axes
        self.show_subtitles = show_subtitles
        self.snapshot_path = snapshot_path

        # set the background color of rendering
        self.window.renderer.set_clear_color(background_color)

        # create subplot viewports. Caveat: [[None] * self.cols] * self.rows is WRONG!!! it's shallow copy
        self.viewports = [[None] * self.cols for _ in range(self.rows)]
        self.viewport_titles = [[None] * self.cols for _ in range(self.rows)]
        self.viewport_legends = [[None] * self.cols for _ in range(self.rows)] # placeholder for legends of each viewport (usually no legend except the semantic plot. legend is set by the PointCloudVis class, and ready() will draw the legend if it's not None)
        for i in range(self.rows):
            for j in range(self.cols):
                self.viewports[i][j] = gui.SceneWidget()
                self.viewports[i][j].scene = rendering.Open3DScene(self.window.renderer)
                self.viewports[i][j].scene.set_background(background_color)
                self.viewports[i][j].scene.show_axes(self.show_axes)
                self.viewport_titles[i][j] = gui.Label("")
                self.viewport_titles[i][j].background_color = gui.Color(*background_color)
                self.viewport_titles[i][j].text_color = gui.Color(0,0,0,1) # black text
                self.viewport_titles[i][j].visible = self.show_subtitles

        def on_mouse(event):
            '''
            set mouse event (to sync camera parameter across all scenes)
            Example: https://github.com/intel-isl/Open3D/blob/master/examples/python/gui/mouse-and-point-coord.py
            Important information:
            1. mouse event: http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.MouseEvent.html
                - MOVE: mouse moving in the window
                - BUTTON_DOWN/BUTTON_UP: button press/release. this applies to all left/middle/right buttons, i.e. pressing any of the three buttons is a DOWN/UP.
                - DRAG: keep pressing and move. Usually for rotate/pan/move, its BUTTON_DOWN -- DRAG ... -- DRAG -- BUTTON_UP.
                - WHEEL: scroll the wheel.
            in Open3D scene, 
                - Left button drag --> rotate camera
                - Right button drag / Ctrl + Left button drag --> pan camera
                - Wheel button drag --> adjust lighting
                - Wheel --> Zoom
            2. three types of event handling: http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html#open3d.visualization.gui.SceneWidget.EventCallbackResult
                - IGNORED: my handler don't handle this. base widget handles it normally
                - HANDLED: my handler handle this. base widget also handles it normally. Useful for augmenting the base functionality
                - CONSUMED: my handler handle this. base widget doesn't see the event. Useful for replacing existing base functionality
            '''
            if self.sync_camera:
                # we notice that rotate & pan must end by a UP, and zoom include WHEEL
                if event.type == gui.MouseEvent.Type.BUTTON_UP or event.type == gui.MouseEvent.Type.WHEEL:
                    # determine which subplot the mouse is in when changing the camera view
                    window_size = self.window.size # window.content_rect is no longer valid after divided into subplot areas
                    subplot_width, subplot_height = window_size.width / self.cols, window_size.height / self.rows
                    row, col = min(int(event.y // subplot_height), self.rows-1), min(int(event.x // subplot_width), self.cols-1) # event.x/y is the mouse location in the window, use min() to avoid out of boundary clicks
                    
                    # sync the camera parameters across all scenes
                    for i in range(self.rows):
                        for j in range(self.cols):
                            self.viewports[i][j].scene.camera.copy_from(self.viewports[row][col].scene.camera)
                            self.viewports[i][j].force_redraw()

                    # augmented functinality is handled. base functionality still works
                    return gui.SceneWidget.EventCallbackResult.HANDLED
            
            return gui.SceneWidget.EventCallbackResult.IGNORED # ignore all other mouse events
        
        def on_key(event):
            '''
            set key event (to save snapshot by press S)
            '''
            if event.key == gui.KeyName.S and event.type == gui.KeyEvent.UP:
                window_size = self.window.size 
                subplot_width, subplot_height = int(window_size.width / self.cols), int(window_size.height / self.rows)
                for i in range(self.rows):
                    for j in range(self.cols): 
                        img = self.app.gui.render_to_image(self.viewports[i][j].scene, subplot_width, subplot_height)
                        o3d.io.write_image(os.path.join(self.snapshot_path, self.viewport_titles[i][j].text + '.png'), img, quality=9)
                # augmented functinality is handled. base functionality still works
                return gui.SceneWidget.EventCallbackResult.HANDLED
            
            return gui.SceneWidget.EventCallbackResult.IGNORED # ignore all other mouse events

        for i in range(self.rows):
            for j in range(self.cols):
                self.viewports[i][j].set_on_mouse(on_mouse)
                self.viewports[i][j].set_on_key(on_key)                    
    
    def set_subplot(self, row, col, title):
        '''
        Set the subplot title and return the subplot handle (a SceneWidget object: http://www.open3d.org/docs/latest/python_api/open3d.visualization.gui.SceneWidget.html) 
        Typical usage:
            p = fig.set_subplot(i,j,'Figure 1a')
            p.scene.add_geometry(name, geometry, material)
            p.setup_camera(field_of_view, model_bounds, center_of_rotation)
        '''
        if row >= self.rows or col >= self.cols:
            self.app.close()
            sys.exit(f"'{self.name}' doesn't have subplot ({row},{col})")

        self.viewport_titles[row][col].text = title
        return Plot3DSubFigure(self, row, col, self.viewports[row][col])

    def ready(self):
        '''
        Mark the figure as ready to be plotted after the subtitles & contents of all subplots are input.
        '''
        # link all children
        for i in range(self.rows):
            for j in range(self.cols):
                self.window.add_child(self.viewports[i][j])
                self.window.add_child(self.viewport_titles[i][j])
                if self.viewport_legends[i][j] is not None: # legend could be None
                    self.window.add_child(self.viewport_legends[i][j])

        # callback to set subplot/subtitle/legend layout
        def on_layout(layout_context):
            full_frame = self.window.content_rect
            origin_x, origin_y = full_frame.x, full_frame.y
            subplot_width, subplot_height = full_frame.width / self.cols, full_frame.height / self.rows
            
            # subplot and subtitle frames
            for i in range(self.rows):
                for j in range(self.cols):
                    # subplot in place
                    if self.plot_boundary:
                        self.viewports[i][j].frame = gui.Rect(int(origin_x + j * subplot_width), int(origin_y + i * subplot_height), int(subplot_width - 1), int(subplot_height - 1))
                    else:
                        self.viewports[i][j].frame = gui.Rect(int(origin_x + j * subplot_width), int(origin_y + i * subplot_height), int(subplot_width + 1), int(subplot_height + 1))

                    # subtitle at the bottom middle of each subplot
                    prefer = self.viewport_titles[i][j].calc_preferred_size(layout_context, gui.Widget.Constraints()) # decide the textbox size based on the given title
                    self.viewport_titles[i][j].frame = gui.Rect(int(origin_x + j * subplot_width + subplot_width/2 - prefer.width/2), int(origin_y + i * subplot_height + subplot_height - prefer.height), prefer.width, prefer.height)

                    # legend at the top right of each subplot
                    if self.viewport_legends[i][j] is not None:
                        prefer = self.viewport_legends[i][j].calc_preferred_size(layout_context, gui.Widget.Constraints()) # decide the grid size based on the given legend
                        # due to the ColorEdit(), the prefer width will be somehow extremely large, so I manually fix the width
                        prefer.width = 180
                        self.viewport_legends[i][j].frame = gui.Rect(int(origin_x + (j+1) * subplot_width - prefer.width), int(origin_y + i * subplot_height), prefer.width, prefer.height)

        # set the layout
        self.window.set_on_layout(on_layout)

    def close(self):
        '''
        Close the figure.
        '''
        self.window.close()
    
'''
Plot functionalities for point cloud data. This is a class with all static methods

[CAVEAT]
For point cloud drawing, if the point cloud has been divided into blocks and loaded by PyTorch dataloader, be aware that the drawing of batch won't make sense. This is because:
    - The dataset is usually made mappable instead of iterable, so the sequence of block doesn't preserve the spatial division.
    - Dataloader has shuffle. Even if you turn off shuffle, the first reason still matters.
'''
class PointCloudVis:
    
    @staticmethod
    def get_default_material_pointcloud():
        '''
        get a default material that can be used to render different geometries
        see Material fields at: http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Material.html. For example, mat.base_color = (0,0,1,1)
        '''
        mat = rendering.Material()
        mat.shader = "defaultUnlit"
        return mat
   
    @staticmethod
    def get_default_material_lineset():
        mat = rendering.Material()
        mat.shader = "unlitLine"
        return mat

    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(i/float(N), 1, np.random.uniform(0.7,1.0)) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc_raw(subfigure_handle, pc_xyzrgb):
        '''
        [Block data] draw points in all blocks using its own point color
            subfigure_handle: Plot3DSubfigure() object
            pc_xyzrgb: (x_global,y_global,z_global,r,g,b), [B,N,6] ndarray, B - num of blocks, N - num of points in a block
        '''
        plot_handle = subfigure_handle.handle
        pc = o3d.geometry.PointCloud()
        pc_xyzrgb = pc_xyzrgb.reshape(-1,pc_xyzrgb.shape[-1]) # flatten all blocks to [B*N,6]
        
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, :3]) # from numpy to o3d format
        if np.max(pc_xyzrgb[:, 3:]) > 1: ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:]/255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:])
        
        plot_handle.scene.add_geometry("Raw Points", pc, PointCloudVis.get_default_material_pointcloud())
        plot_handle.setup_camera(60, plot_handle.scene.bounding_box, plot_handle.scene.bounding_box.get_center())

    @staticmethod
    def draw_pc_by_block(subfigure_handle, pc_xyzrgb, show_bbox=True, bbox_color='color'):
        '''    
        [Block data] draw points colored by block (each block displayed with a different color)
            subfigure_handle: Plot3DSubfigure() object
            pc_xyzrgb: (x_global,y_global,z_global,r,g,b), [B,N,6] ndarray, B - num of blocks, N - num of points in a block
            show_bbox: switch to draw bbox of each block
            bbox_color: 'black' to draw black boundary; 'color' to draw the same as instance color
        '''
        plot_handle = subfigure_handle.handle
        pc = o3d.geometry.PointCloud()
        
        # assign per-block color
        num_blocks = pc_xyzrgb.shape[0]
        colors = PointCloudVis.random_colors(num_blocks)
        for block_i in range(num_blocks):
            pc_xyzrgb[block_i,:,3:] = colors[block_i]
        
        # draw bbox for each block
        if show_bbox:
            for block_i in range(num_blocks):
                xyz = o3d.utility.Vector3dVector(pc_xyzrgb[block_i, :, :3])  # all points belong to this block
                bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(xyz)
                bbox_wireframe = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                if bbox_color == 'black':
                    bbox_wireframe.paint_uniform_color((0,0,0))
                else:
                    bbox_wireframe.paint_uniform_color(colors[block_i])
                plot_handle.scene.add_geometry(f"Block {block_i}", bbox_wireframe, PointCloudVis.get_default_material_lineset())

        pc_xyzrgb = pc_xyzrgb.reshape(-1,pc_xyzrgb.shape[-1]) # flatten all blocks
        
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, :3]) # from numpy to o3d format
        if np.max(pc_xyzrgb[:, 3:]) > 1: ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:]/255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:])
        
        plot_handle.scene.add_geometry("Points by Block", pc, PointCloudVis.get_default_material_pointcloud())
        plot_handle.setup_camera(60, plot_handle.scene.bounding_box, plot_handle.scene.bounding_box.get_center())

    @staticmethod
    def draw_pc_by_semins(subfigure_handle, pc_xyzrgbsemins, color_code='instance',sem_dict=None, show_legend=False, show_bbox=True, bbox_axis_align=True, bbox_color='black', show_instance_label=False):
        '''
        [Block data] draw points colored by semantic/instance label
            - 'semantic' mode: each semantic class displayed in a different color, option to show legend aside. -1 means not a class, colored as black
            - 'instance' mode: each instance displayed in a different color, options to display bbox (axis aligned/oriented), to show bbox boundary in black/color, to show instance label "ID: class". -1 menas not an instance, colored as black

            subfigure_handle: Plot3DSubfigure() object
            pc_xyzrgbsemins: (x_global,y_global,z_global,r,g,b,sem,ins), [B,N,8] ndarray, B - num of blocks, N - num of points in a block
            color_code: switch to plot by semantic/instance
            sem_dict: dictionary {semantic_ID: semantic_class_name} or just a list with [semantic_class_name]. This could be used in both mode to display class legends or instance labels

            Options for color_code = 'semantic':
                - show_legend: switch to show semantic legend. To really display the legend, remember to call figure.show_legend() on the higher-level figure handle
            Options for color_code = 'instance':
                - show_bbox: switch to show bbox.
                - bbox_axis_align: switch to plot axis align bbox (usually for rooms like S3DIS) or oriented bbox.
                - bbox_color: 'black' to draw black boundary; 'color' to draw the same as instance color
                - show_instance_label: switch to show {ID: class} label on bbox. CAVEAT: only show labels on a (1,1) viewport layout! with multiple subplot the label display is incorrect.
        '''
        plot_handle = subfigure_handle.handle
        sem_labels, ins_labels = pc_xyzrgbsemins[:,:,-2].flatten(), pc_xyzrgbsemins[:,:,-1].flatten()
        sem_keys, ins_keys = np.unique(sem_labels), np.unique(ins_labels)
        sem_keys, ins_keys = sem_keys[sem_keys > -1], ins_keys[ins_keys > -1] # remove non-class/non-instance labels
        sem_colors = PointCloudVis.random_colors(len(sem_keys)) 
        ins_colors = PointCloudVis.random_colors(len(ins_keys))
        print('ins_keys', len(ins_keys))
        
        print(f"Total: {len(sem_keys)} classes, {len(ins_keys)} instances")

        # plot semantic segmentation
        pc = o3d.geometry.PointCloud()

        pc_xyzrgb = pc_xyzrgbsemins[:,:,:6].reshape(-1,6) # flatten
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, :3]) # from numpy to o3d format
        
        non_plot_color = (0.7,0.7,0.7) # for non-class/non-instance points
        if color_code == 'semantic':
            # first assign black to no-class points
            pc_xyzrgb[np.argwhere(sem_labels == -1)[:, 0],3:] = non_plot_color
            # assign per-class color            
            for id, sem in enumerate(sem_keys):
                sem_ind = np.argwhere(sem_labels == sem)[:, 0]
                pc_xyzrgb[sem_ind,3:] = sem_colors[id]

            if sem_dict is not None and show_legend == True:
                vgrid = gui.VGrid(2) 
                vgrid.background_color = gui.Color(1,1,1)
                for i, sem in enumerate(sem_keys):
                    class_name = sem_dict[int(sem)]
                    class_color = sem_colors[i]
                    field1 = gui.Label(class_name)
                    field1.text_color = gui.Color(0,0,0)
                    field2 = gui.ColorEdit()
                    field2.color_value = gui.Color(*class_color)
                    vgrid.add_child(field1)
                    vgrid.add_child(field2)
                
                figure_handle = subfigure_handle.parent 
                figure_handle.viewport_legends[subfigure_handle.row][subfigure_handle.col] = vgrid

        elif color_code == 'instance':
            # first assign black to no-instance points
            pc_xyzrgb[np.argwhere(ins_labels == -1)[:, 0],3:] = non_plot_color
            # assign per-instance color
            for id, ins in enumerate(ins_keys):
                ins_ind = np.argwhere(ins_labels == ins)[:, 0]
                pc_xyzrgb[ins_ind,3:] = ins_colors[id]

                # draw bbox for each instance
                if show_bbox:
                    xyz = o3d.utility.Vector3dVector(pc_xyzrgb[ins_ind, :3])  # all points belong to this instance
                    if bbox_axis_align:
                        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(xyz)
                        bbox_wireframe = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                    else:
                        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(xyz)
                        bbox_wireframe = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)

                    if bbox_color == 'black':
                        bbox_wireframe.paint_uniform_color((0,0,0))
                    else:
                        bbox_wireframe.paint_uniform_color(ins_colors[id])
                    
                    # plot_handle.scene.add_geometry(f"Instance BBox {id}", bbox_wireframe, PointCloudVis.get_default_material_lineset())
                    # after 0.13.0, we can directly draw bbox instead of converting to lineset
                    plot_handle.scene.add_geometry(f"Instance BBox {id}", bbox, PointCloudVis.get_default_material_lineset())

                    if show_instance_label:
                        
                        # add text label {ID: class} to each instance
                        sem_class = np.unique(sem_labels[ins_ind]).astype(int)
                        if len(sem_class) > 1:
                            print(f"[ERROR] Instance {id} doesn't map to a unique semantic class")
                            continue
                        sem_class = sem_class[0]
                        if sem_dict is not None:
                            sem_class = sem_dict[sem_class]
                        plot_handle.add_3d_label(bbox.get_center(), f"{id}: {sem_class}")
            if show_instance_label:
                print("For the instance text label to display properly, make sure there is no other subplots in this window!")

        pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:])
        plot_handle.scene.add_geometry("Points by SemIns", pc, PointCloudVis.get_default_material_pointcloud())
        plot_handle.setup_camera(60, plot_handle.scene.bounding_box, plot_handle.scene.bounding_box.get_center())    
    
    
    @staticmethod
    def draw_pc_by_lidar(subfigure_handle, pc_xyzrgblidar, lidar_positions, show_rays=False, show_rays_lidar=None):
        '''    
        [Flatten data] draw points colored by LiDAR (points sensed by each LiDAR displayed with a different color)
            subfigure_handle: Plot3DSubfigure() object
            pc_xyzrgblidar: (x_global,y_global,z_global,r,g,b, lidar_id), [N,7] ndarray
            lidar_positions: (lidar_x, lidar_y, lidar_z), [L, 3], LiDAR positions
            show_rays: switch to display LiDAR rays
            show_rays_lidar: list(LiDAR IDs that will show rays). Default value is None to show all lidar rays
        '''
        plot_handle = subfigure_handle.handle
        pc = o3d.geometry.PointCloud()
        
        num_lidars = lidar_positions.shape[0]
        colors = PointCloudVis.random_colors(num_lidars)
        for lidar_i in range(num_lidars):
            lidar_ind = np.argwhere(pc_xyzrgblidar[:,-1] == lidar_i)[:, 0]
            print(f"LiDAR {lidar_i} senses {len(lidar_ind)} points")
            if len(lidar_ind) == 0:
                continue

            # assign per-lidar color
            pc_xyzrgblidar[lidar_ind, 3:6] = colors[lidar_i]

            # plot LiDARs
            sphere_lidar = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.1)
            sphere_lidar.paint_uniform_color(colors[lidar_i])
            sphere_lidar.translate(lidar_positions[lidar_i])
            plot_handle.scene.add_geometry(f"LiDAR {lidar_i}", sphere_lidar, PointCloudVis.get_default_material_pointcloud())
            plot_handle.add_3d_label(lidar_positions[lidar_i], f"LiDAR {lidar_i}")

            # plot rays
            if show_rays:
                if show_rays_lidar is None or lidar_i in show_rays_lidar:
                    line_pts = np.concatenate((lidar_positions[lidar_i].reshape(1,-1), pc_xyzrgblidar[lidar_ind,:3]), axis=0)
                    line_indices = [[0, pt+1] for pt in range(len(lidar_ind))] # from lidar pos to all its points
                    rays = o3d.geometry.LineSet()
                    rays.points = o3d.utility.Vector3dVector(line_pts)
                    rays.lines = o3d.utility.Vector2iVector(line_indices)
                    rays.paint_uniform_color(colors[lidar_i])
                    mat = PointCloudVis.get_default_material_lineset()
                    mat.line_width = 0.1
                    plot_handle.scene.add_geometry(f"LiDAR {lidar_i} Rays", rays, mat)

        pc.points = o3d.utility.Vector3dVector(pc_xyzrgblidar[:, :3]) # from numpy to o3d format
        if np.max(pc_xyzrgblidar[:, 3:6]) > 1: ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgblidar[:, 3:6]/255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgblidar[:, 3:6])
        
        plot_handle.scene.add_geometry("Points by LiDAR", pc, PointCloudVis.get_default_material_pointcloud())
        plot_handle.setup_camera(60, plot_handle.scene.bounding_box, plot_handle.scene.bounding_box.get_center())
