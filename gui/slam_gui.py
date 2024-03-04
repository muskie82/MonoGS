import pathlib
import threading
import time
from datetime import datetime

import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F
from OpenGL import GL as gl

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gui.gl_render import util, util_gau
from gui.gl_render.render_ogl import OpenGLRenderer
from gui.gui_utils import (
    GaussianPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
)
from utils.camera_utils import Camera
from utils.logging_utils import Log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.model_dict = {}

        self.init_widget()

        self.q_main2vis = None
        self.gaussian_cur = None
        self.pipe = None
        self.background = None

        self.init = False
        self.kf_window = None
        self.render_img = None

        if params_gui is not None:
            self.background = params_gui.background
            self.gaussian_cur = params_gui.gaussians
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.pipe = params_gui.pipe

        self.gaussian_nums = []

        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    def init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "MonoGS", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.button = gui.ToggleSwitch("Resume/Pause")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Camera follow options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))

        ## Jump to the camera viewpoint
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(gui.Label("Viewpoint list"))
        combo_tile.add_child(self.combo_kf)
        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.kf_window_chbox = gui.Checkbox("Active window")
        self.kf_window_chbox.set_on_checked(self._on_kf_window_chbox)
        chbox_tile_3dobj.add_child(self.kf_window_chbox)
        self.panel.add_child(chbox_tile_3dobj)

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = False
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        self.panel.add_child(gui.Label("Rendering options"))
        chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        chbox_tile_geometry.add_child(self.depth_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        chbox_tile_geometry.add_child(self.opacity_chbox)

        self.time_shader_chbox = gui.Checkbox("Time Shader")
        self.time_shader_chbox.checked = False
        chbox_tile_geometry.add_child(self.time_shader_chbox)

        self.elipsoid_chbox = gui.Checkbox("Elipsoid Shader")
        self.elipsoid_chbox.checked = False
        chbox_tile_geometry.add_child(self.elipsoid_chbox)

        self.panel.add_child(chbox_tile_geometry)

        slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Scale (0-1)")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.scaling_slider)
        self.panel.add_child(slider_tile)

        # screenshot buttom
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        ## Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)
        self.output_info = gui.Label("Number of Gaussians: ")
        tab_info.add_child(self.output_info)

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Input Color/Depth"))
        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(-4)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], gt=False, size=0.01):
        W2C = (
            getWorld2View2(camera.R_gt, camera.T_gt)
            if gt
            else getWorld2View2(camera.R, camera.T)
        )
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.7
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_combo_model(self, new_val, new_idx):
        model_idx = self.model_dict[new_val]
        self.global_map.active_map_idx = model_idx

    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_axis_chbox(self, is_checked):
        name = "axis"
        if is_checked:
            self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, self.axis, self.lit_geo)
        else:
            self.widget3d.scene.remove_geometry(name)

    def _on_kf_window_chbox(self, is_checked):
        if self.kf_window is None:
            return
        edge_cnt = 0
        for key in self.kf_window.keys():
            for kf_idx in self.kf_window[key]:
                name = "kf_edge_{}".format(edge_cnt)
                edge_cnt += 1
                if "keyframe_{}".format(key) not in self.frustum_dict.keys():
                    continue
                test1 = self.frustum_dict["keyframe_{}".format(key)].view_dir[1]
                kf = self.frustum_dict["keyframe_{}".format(kf_idx)].view_dir[1]
                points = [test1, kf]
                lines = [[0, 1]]
                colors = [[0, 1, 0]]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                if is_checked:
                    self.widget3d.scene.remove_geometry(name)
                    self.widget3d.scene.add_geometry(name, line_set, self.lit)
                else:
                    self.widget3d.scene.remove_geometry(name)

    def _on_button(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.button.is_on
        self.q_vis2main.put(packet)

    def _on_slider(self, value):
        packet = self.prepare_viz2main_packet()
        self.q_vis2main.put(packet)

    def _on_render_btn(self):
        packet = Packet_vis2main()
        packet.flag_nextbatch = True
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui.png", img)
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    def add_ids(self):
        indices = (
            torch.unique(self.gaussian_cur.unique_kfIDs).cpu().numpy().astype(int)
        ).tolist()
        for idx in indices:
            if idx in self.gaussian_id_dict.keys():
                continue

            self.gaussian_id_dict[idx] = 0
            self.combo_gaussian_id.add_item(str(idx))

    def receive_data(self, q):
        if q is None:
            return

        gaussian_packet = get_latest_queue(q)
        if gaussian_packet is None:
            return

        if gaussian_packet.has_gaussians:
            self.gaussian_cur = gaussian_packet
            self.output_info.text = "Number of Gaussians: {}".format(
                self.gaussian_cur.get_xyz.shape[0]
            )
            self.init = True

        if gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        if gaussian_packet.keyframe is not None:
            name = "keyframe_{}".format(gaussian_packet.keyframe.uid)
            frustum = self.add_camera(
                gaussian_packet.keyframe, name=name, color=[0, 0, 1]
            )

        if gaussian_packet.keyframes is not None:
            for keyframe in gaussian_packet.keyframes:
                name = "keyframe_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[0, 0, 1])

        if gaussian_packet.kf_window is not None:
            self.kf_window = gaussian_packet.kf_window
            self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)

        if gaussian_packet.gtcolor is not None:
            rgb = torch.clamp(gaussian_packet.gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if gaussian_packet.gtdepth is not None:
            depth = gaussian_packet.gtdepth
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=5.0, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(depth)
            self.in_depth_widget.update_image(rgb)

        if gaussian_packet.finish:
            Log("Received terminate signal", tag="GUI")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    @staticmethod
    def depth_to_normal(points, k=3, d_min=1e-3, d_max=10.0):
        k = (k - 1) // 2
        # points: (B, 3, H, W)
        b, _, h, w = points.size()
        points_pad = F.pad(
            points, (k, k, k, k), mode="constant", value=0
        )  # (B, 3, k+H+k, k+W+k)
        if d_max is not None:
            valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
                points_pad[:, 2:, :, :] < d_max
            )  # (B, 1, k+H+k, k+W+k)
        else:
            valid_pad = points_pad[:, 2:, :, :] > d_min
        valid_pad = valid_pad.float()

        # vertical vector (top - bottom)
        vec_vert = (
            points_pad[:, :, :h, k : w + k]
            - points_pad[:, :, 2 * k : h + (2 * k), k : w + k]
        )

        # horizontal vector (left - right)
        vec_hori = (
            points_pad[:, :, k : h + k, :w]
            - points_pad[:, :, k : h + k, 2 * k : w + (2 * k)]
        )

        # valid_mask
        valid_mask = (
            valid_pad[:, :, k : h + k, k : w + k]
            * valid_pad[:, :, :h, k : w + k]
            * valid_pad[:, :, 2 * k : h + (2 * k), k : w + k]
            * valid_pad[:, :, k : h + k, :w]
            * valid_pad[:, :, k : h + k, 2 * k : w + (2 * k)]
        )
        valid_mask = valid_mask > 0.5

        # get cross product (B, 3, H, W)
        cross_product = -torch.linalg.cross(vec_vert, vec_hori, dim=1)
        normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
        return normal, valid_mask

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam

    def rasterise(self, current_cam):
        if (
            self.time_shader_chbox.checked
            and self.gaussian_cur is not None
            and type(self.gaussian_cur) == GaussianPacket
        ):
            features = self.gaussian_cur.get_features.clone()
            kf_ids = self.gaussian_cur.unique_kfIDs.float()
            rgb_kf = imgviz.depth2rgb(
                kf_ids.view(-1, 1).cpu().numpy(), colormap="jet", dtype=np.float32
            )
            alpha = 0.1
            self.gaussian_cur.get_features = alpha * features + (
                1 - alpha
            ) * torch.from_numpy(rgb_kf).to(features.device)
            rendering_data = render(
                current_cam,
                self.gaussian_cur,
                self.pipe,
                self.background,
                self.scaling_slider.double_value,
            )
            self.gaussian_cur.get_features = features
        else:
            rendering_data = render(
                current_cam,
                self.gaussian_cur,
                self.pipe,
                self.background,
                self.scaling_slider.double_value,
            )
        return rendering_data

    def render_o3d_image(self, results, current_cam):
        if self.depth_chbox.checked:
            depth = results["depth"]
            depth = depth[0, :, :].detach().cpu().numpy()
            max_depth = np.max(depth)
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=max_depth, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(depth)

        elif self.opacity_chbox.checked:
            opacity = results["opacity"]
            opacity = opacity[0, :, :].detach().cpu().numpy()
            max_opacity = np.max(opacity)
            opacity = imgviz.depth2rgb(
                opacity, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity = torch.from_numpy(opacity)
            opacity = torch.permute(opacity, (2, 0, 1)).float()
            opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(opacity)

        elif self.elipsoid_chbox.checked:
            if self.gaussian_cur is None:
                return
            glfw.poll_events()
            gl.glClearColor(0, 0, 0, 1.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w = int(self.window.size.width * self.widget3d_width_ratio)
            glfw.set_window_size(self.window_gl, w, self.window.size.height)
            self.g_camera.fovy = current_cam.FoVy
            self.g_camera.update_resolution(self.window.size.height, w)
            self.g_renderer.set_render_reso(w, self.window.size.height)
            frustum = create_frustum(
                np.linalg.inv(cv_gl @ self.widget3d.scene.camera.get_view_matrix())
            )

            self.g_camera.position = frustum.eye.astype(np.float32)
            self.g_camera.target = frustum.center.astype(np.float32)
            self.g_camera.up = frustum.up.astype(np.float32)

            self.gaussians_gl.xyz = self.gaussian_cur.get_xyz.cpu().numpy()
            self.gaussians_gl.opacity = self.gaussian_cur.get_opacity.cpu().numpy()
            self.gaussians_gl.scale = self.gaussian_cur.get_scaling.cpu().numpy()
            self.gaussians_gl.rot = self.gaussian_cur.get_rotation.cpu().numpy()
            self.gaussians_gl.sh = self.gaussian_cur.get_features.cpu().numpy()[:, 0, :]

            self.update_activated_renderer_state(self.gaussians_gl)
            self.g_renderer.sort_and_update(self.g_camera)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.g_renderer.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            cv2.flip(img, 0, img)
            render_img = o3d.geometry.Image(img)
            glfw.swap_buffers(self.window_gl)
        else:
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)
        return render_img

    def render_gui(self):
        if not self.init:
            return
        current_cam = self.get_current_cam()
        results = self.rasterise(current_cam)
        if results is None:
            return
        self.render_img = self.render_o3d_image(results, current_cam)
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.render_gui()

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                Log("Closing Visualization", tag="GUI")
                break

            def update():
                if self.step % 3 == 0:
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
