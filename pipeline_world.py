from pathlib import Path

from pipeline_common import Operator, Pipeline

import h5py
import matplotlib.pyplot as plt
import numpy as np


FIGURE_DPI = 120

"""
Some guidelines:
- self.spec & self.context are at your disposal
- outputs (both the set of outputs and their contents) should only ever depend on the spec and explicitly stated inputs
- all inputs and outputs must be distinct -- an operator cannot modify files in place

For operator development, you can use
    CACHEABLE = False
"""


class GeoStep(Operator):
    NAME = "geo"
    VERSION = 9
    TRIGGERS = ["geo:plot"]

    def get_inputs(self):
        deps = set()

        if "heightmap" in self.spec:
            deps.add(self.context.spec_dir / self.spec["heightmap"])

        if "midpoint_displacement_fractal" in self.spec and isinstance(self.spec["midpoint_displacement_fractal"]["primary_scale"], str):
            deps.add(self.spec["midpoint_displacement_fractal"]["primary_scale"])

        if "midpoint_displacement_fractal" in self.spec and isinstance(self.spec["midpoint_displacement_fractal"]["roughness"], str):
            deps.add(self.spec["midpoint_displacement_fractal"]["roughness"])

        return deps

    def get_outputs(self):
        return {"terrain.h5"}

    def run(self, input_paths, output_paths):
        spec = self.spec
        context = self.context

        if "heightmap" in spec:
            # Just load heightmap and produce HDF5 output

            heightmap_path = context.spec_dir / spec["heightmap"]

            if heightmap_path.suffix == ".hmap":
                from worldlib.exch_hmap import read_hmap

                with open(heightmap_path, "rb") as f:
                    height, xyres = read_hmap(f)
            else:
                raise Exception()

            with h5py.File(output_paths["terrain.h5"], "w") as f:
                f['heightmap'] = height.astype(np.float32)
                f.attrs['heightmap_resolution'] = xyres
        elif "midpoint_displacement_fractal" in spec:
            options = spec["midpoint_displacement_fractal"]

            # TODO: replace with procgenlib
            from worldgenlib.terrain_gen import random_midpoint_displacement_fractal

            rng = np.random.default_rng(seed=options["seed"])

            if isinstance(options["primary_scale"], str):
                primary_scale = np.load(input_paths[options["primary_scale"]])
            else:
                primary_scale = options["primary_scale"]

            if isinstance(options["roughness"], str):
                roughness = np.load(input_paths[options["roughness"]])
            else:
                roughness = options["roughness"]

            height = random_midpoint_displacement_fractal(rng=rng,
                                                          num_squares=options["num_squares"],
                                                          square_size=options["square_size"],
                                                          base_level=options["base_level"],
                                                          primary_scale=primary_scale,
                                                          roughness=roughness)

            with h5py.File(output_paths["terrain.h5"], "w") as f:
                f['heightmap'] = height.astype(np.float32)
                f.attrs['heightmap_resolution'] = options["cell_size_meters"]
                #f['water_column'] = water_depth
        else:
            raise Exception()


class GeoPlotStep(Operator):
    NAME = "geo:plot"
    VERSION = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input = self.spec.get("terrain", "terrain.h5")
        self.output = str(Path(self.input).with_suffix(".png"))

    def get_inputs(self):
        return {self.input}

    def get_outputs(self):
        return {self.output}

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths[self.input], "r") as f:
            wh_xyres = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        extent = tuple(np.array((-0.5, heightmap.shape[0] - 0.5, -0.5, heightmap.shape[1] - 0.5)) * wh_xyres)
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(heightmap.T, origin='lower', extent=extent, cmap='gist_earth', vmin=-200, vmax=2000)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Terrain base height ({heightmap.shape[0]}x{heightmap.shape[1]} cells, {wh_xyres}x{wh_xyres}m each)")
        ax.set_xlabel("Meters")
        ax.set_ylabel("Meters")
        fig.savefig(output_paths[self.output], bbox_inches="tight", dpi=FIGURE_DPI)


class HydroSimStep(Operator):
    NAME = "hydro_sim"
    VERSION = 3

    def get_inputs(self):
        return {"terrain.h5"}

    def get_outputs(self):
        return {"water.h5", "water_sim_convergence.png"}

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain.h5"], "r") as f:
            wh_xyres = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        from worldgenlib import water

        # ugly hack
        kwargs = {**self.spec}#; del kwargs["level_reduction"]

        res = water.simulate(heightmap=heightmap,
                    #viscosity=spec["viscosity"],
                    #rainfall=spec["rainfall"],
                    max_flow_change=1e-3,
                    max_level_change=1e-3,
                    verbose=True,
                    **kwargs
                    )

        with h5py.File(output_paths["water.h5"], "w") as f:
            f["water_surface_level"] = res.surface_level.astype(np.float32)
            f["water_x_flow"] = res.x_flow.astype(np.float32)
            f["water_y_flow"] = res.y_flow.astype(np.float32)
            f["water_surface_level_changelog"] = res.surface_level_changelog.astype(np.float32)
            f["water_x_flow_changelog"] = res.x_flow_changelog.astype(np.float32)
            f["water_y_flow_changelog"] = res.y_flow_changelog.astype(np.float32)
            f["water_volume_changelog"] = res.volume_changelog.astype(np.float32)

        fig, axs = plt.subplots(figsize=(12, 6), ncols=2, nrows=2)
        ax = axs[0, 0]
        # surface level convergence
        ax.semilogy(res.surface_level_changelog)
        ax.grid()
        ax.set_title("Surface level change rate")
        ax.set_xlabel("Simulation step")
        #x/y flow convergence
        ax = axs[0, 1]
        ax.semilogy(res.x_flow_changelog, label="x")
        ax.semilogy(res.y_flow_changelog, label="y")
        ax.grid()
        ax.legend()
        ax.set_title("Flow change rate")
        ax.set_xlabel("Simulation step")
        # Volume evolution
        ax = axs[1, 0]
        ax.plot(res.volume_changelog)
        ax.grid()
        ax.set_title("Total volume")
        ax.set_ylabel("Water volume")
        ax.set_xlabel("Simulation step")
        # Time delta evolution
        ax = axs[1, 1]
        ax.plot(res.dt_changelog)
        ax.grid()
        ax.set_title("Time step")
        ax.set_ylabel("dt")
        ax.set_xlabel("Simulation step")
        fig.tight_layout()
        fig.savefig(output_paths["water_sim_convergence.png"], bbox_inches="tight", dpi=FIGURE_DPI)


class HydroAdjustStep(Operator):
    NAME = "hydro_adj"
    VERSION = 3

    def get_inputs(self):
        return {"terrain.h5", "water.h5"}

    def get_outputs(self):
        return {"terrain_adj.h5", "water_adj.h5"}.union({"erosion.png"} if "erosion" in self.spec else set())

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain.h5"], "r") as f:
            wh_xyres = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        with h5py.File(input_paths["water.h5"], "r") as f:
            surface_level = f["water_surface_level"][()]
            x_flow = f["water_x_flow"][()]
            y_flow = f["water_y_flow"][()]

        from worldgenlib import water

        if "erosion" not in self.spec:
            new_heightmap = heightmap
        else:
            resx = water.simulate(
                heightmap=heightmap,
                init=water.Init(surface_level=surface_level, x_flow=x_flow, y_flow=y_flow),
                # viscosity=0.1, # FIXME
                # rainfall=0.01, # FIXME
                viscosity=self.spec["erosion"]["viscosity"],
                rainfall=self.spec["erosion"]["rainfall"],

                max_iters=self.spec["erosion"]["iterations"],
                erosion_strength=self.spec["erosion"]["strength"],
                erosion_sideways_coeff=self.spec["erosion"]["sideways_coeff"],

                max_flow_change=1e-3,
                max_level_change=1e-3,
            )
            new_heightmap, surface_level = resx.heightmap, resx.surface_level

            extent = tuple(np.array((-0.5, heightmap.shape[0] - 0.5, -0.5, heightmap.shape[1] - 0.5)) * wh_xyres)
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow((new_heightmap - heightmap).T, origin='lower', extent=extent)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Erosion amount")
            ax.set_xlabel("Meters")
            ax.set_ylabel("Meters")
            fig.savefig(output_paths["erosion.png"], bbox_inches="tight", dpi=FIGURE_DPI)

        surface_level -= self.spec["level_reduction"]

        underground_level = water.adjust_underground_levels(
                    surface_level,
                    new_heightmap,
                    # threshold of water depth to not be discarded as dry surface
                    threshold=self.spec["depth_threshold"]
                    )

        with h5py.File(output_paths["terrain_adj.h5"], "w") as f:
            f.attrs['heightmap_resolution'] = wh_xyres
            f['heightmap'] = new_heightmap

        with h5py.File(output_paths["water_adj.h5"], "w") as f:
            f["water_surface_level"] = surface_level
            f["water_underground_level"] = underground_level.astype(np.float32)
            f["water_x_flow"] = x_flow
            f["water_y_flow"] = y_flow


class HydroPlotsStep(Operator):
    NAME = "hydro_plots"
    VERSION = 9

    def get_inputs(self):
        return {"terrain_adj.h5", "water_adj.h5"}

    def get_outputs(self):
        return {"terrain_adj.png",
                "water_streamplot.png",
                "water_height.png",
                "water_velocity.png",
                "water_underground.png",
                }

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain_adj.h5"], "r") as f:
            wh_xyres = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        with h5py.File(input_paths["water_adj.h5"], "r") as f:
            surface_level = f["water_surface_level"][()]
            underground_level = f["water_underground_level"][()]
            x_flow = f["water_x_flow"][()]
            y_flow = f["water_y_flow"][()]

        extent = tuple(np.array((-0.5, heightmap.shape[0] - 0.5, -0.5, heightmap.shape[1] - 0.5)) * wh_xyres)

        # Terrain stuff

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(heightmap.T, origin='lower', extent=extent, cmap='gist_earth', vmin=-200, vmax=2000)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Terrain base height ({heightmap.shape[0]}x{heightmap.shape[1]} cells, {wh_xyres}x{wh_xyres}m each)")
        ax.set_xlabel("Meters"); ax.set_ylabel("Meters")
        fig.savefig(output_paths["terrain_adj.png"], bbox_inches="tight", dpi=FIGURE_DPI)

        # Water stuff

        cols, rows = heightmap.shape

        h = np.maximum(heightmap, surface_level)
        # h = surface_level - heightmap
        # h = heightmap
        # h = surface_level
        # speed_x = (np.minimum(0, -x_flow[0:cols, :]) + np.maximum(0, x_flow[1:cols + 1, :])) / 2
        # speed_y = (np.minimum(0, -y_flow[:, 0:rows]) + np.maximum(0, y_flow[:, 1:rows + 1])) / 2
        speed_x = (x_flow[0:cols, :] + x_flow[1:cols + 1, :]) / 2
        speed_y = (y_flow[:, 0:rows] + y_flow[:, 1:rows + 1]) / 2
        speed = np.sqrt(np.power(speed_x, 2) + np.power(speed_y, 2))

        #CLASSIC:
        # speed = np.sqrt(np.power(horz_flow[:, 0:N] + horz_flow[:, 1:N+1], 2) + np.power(vert_flow[0:M, :] + vert_flow[1:M+1, :], 2))

        speed[np.less(surface_level - heightmap, 0.01)] = 0#np.nan
        # print(np.amax(h))

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(h.T, origin="lower", extent=extent)#, cmap="YlGn")

        # if cols < 80:
        #     ax.set_xticks(np.arange(cols))
        # if rows < 80:
        #     ax.set_yticks(np.arange(rows))

        # Streamlines
        y, x = np.arange(rows) * wh_xyres, np.arange(cols) * wh_xyres
        lw = 20 * speed / np.amax(speed) # RuntimeWarning: invalid value encountered in true_divide

        # Blur line width field per https://stackoverflow.com/a/65804973
        a = lw
        kernel = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, a)
        a = np.apply_along_axis(lambda y: np.convolve(y, kernel, mode='same'), 1, a)
        lw = a

        ax.streamplot(x, y, (x_flow[0:cols, :] + x_flow[1:cols+1, :]).T, (y_flow[:, 0:rows] + y_flow[:, 1:rows + 1]).T,
                    color='w', linewidth=lw.T, density=5)

        ax.set_title(f"Relative water velocity (max: {np.amax(speed):.3g})")
        ax.set_xlabel("Meters"); ax.set_ylabel("Meters")
        fig.colorbar(im, ax=ax)
        fig.savefig(output_paths["water_streamplot.png"], bbox_inches="tight", dpi=FIGURE_DPI)

        fig, axs = plt.subplots(figsize=(16, 8), ncols=2)
        im = axs[0].imshow(np.maximum(surface_level - heightmap, 0).T, origin="lower", extent=extent)
        fig.colorbar(im, ax=axs[0])
        axs[0].set_title("Water height above surface")
        axs[0].set_xlabel("Meters"); axs[0].set_ylabel("Meters")
        im = axs[1].imshow(np.clip(surface_level - heightmap, 0, 10).T, origin="lower", extent=extent)
        fig.colorbar(im, ax=axs[1])
        axs[1].set_title("Water height above surface (clamped)")
        axs[1].set_xlabel("Meters"); axs[1].set_ylabel("Meters")
        fig.savefig(output_paths["water_height.png"], bbox_inches="tight", dpi=FIGURE_DPI)

        fig, ax = plt.subplots(figsize=(16, 8))
        speed_rgb = np.zeros((cols, rows, 3))
        speed_rgb[:,:,0] = 0.5 + 0.5 * (speed_x / np.percentile(np.abs(speed_x), 95))
        speed_rgb[:,:,1] = 0.5 + 0.5 * (speed_y / np.percentile(np.abs(speed_y), 95))
        speed_rgb[:,:,2] = 0.5

        ax.imshow(np.swapaxes(np.clip(speed_rgb, 0, 1), 0, 1))
        ax.set_title("Water movement velocity")
        fig.savefig(output_paths["water_velocity.png"], bbox_inches="tight", dpi=FIGURE_DPI)

        # Underground water
        vmin, vmax = np.amin(surface_level), np.amax(surface_level)
        fig, axs = plt.subplots(figsize=(16, 8), ncols=2)
        im = axs[0].imshow(surface_level.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        im = axs[1].imshow(underground_level.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        axs[0].set_title("Surface level pre-thresholding")
        axs[1].set_title("Underground level")
        axs[0].set_xlabel("Meters"); axs[0].set_ylabel("Meters")
        axs[1].set_xlabel("Meters"); axs[1].set_ylabel("Meters")
        fig.savefig(output_paths["water_underground.png"], bbox_inches="tight", dpi=FIGURE_DPI)


class PlyExportStep(Operator):
    NAME = "ply_export"
    VERSION = 2

    def get_inputs(self):
        return {"terrain_adj.h5", "water_adj.h5"}

    def get_outputs(self):
        return {"world.ply"}

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain_adj.h5"], "r") as f:
            heightmap_resolution = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        with h5py.File(input_paths["water_adj.h5"], "r") as f:
            surface_level = f["water_surface_level"][()]
            underground_level = f["water_underground_level"][()]
            x_flow = f["water_x_flow"][()]
            y_flow = f["water_y_flow"][()]

        from outputs.heightmap_ply import tet_ply

        water_column = (underground_level - heightmap)

        ply = tet_ply(heightmap=heightmap,
                      water_column=water_column,
                      heightmap_resolution=heightmap_resolution,
                      h_scale=1,
                      text=False,
                      byte_order='<'
                      )
        ply.write(output_paths["world.ply"])


class BlenderRenderStep(Operator):
    NAME = "blender_render"
    VERSION = 2

    def get_inputs(self):
        return {"world.ply"}

    def get_outputs(self):
        return {"world_render.jpg"}

    def run(self, input_paths, output_paths):
        import subprocess

        # shameless theft from https://github.com/j2me-preservation/MascotCapsule/blob/master/tools/render_obj.py
        def render_ply(plyfile, output_path):
            script_path = Path(__file__).resolve().parent / "blender_importscript292.py"
            assert script_path.is_file()    # blender will helpfully continue even if this critical file is missing

            subprocess.check_call([
                'blender',
                '--background',
                "--engine", "CYCLES",       # only Cycles rendering works without OpenGL
                "-noaudio",                 # reduce stderr clutter (we don't need no sound)
                '--python', str(script_path),
                '--render-output', '//' + str(output_path),
                '--render-format', 'JPEG',
                '--use-extension', '1',
                '-f', '0',
                '--', plyfile
            ])

        import shutil, tempfile
        path = tempfile.mktemp()
        render_ply(input_paths["world.ply"], path)
        shutil.move(path + "0000.jpg", output_paths["world_render.jpg"])


class PerlinOperator(Operator):
    NAME = "perlin"
    VERSION = 4

    def get_inputs(self):
        return set()

    def get_outputs(self):
        return {"$output", "$output.png"}

    def run(self, input_paths, output_paths):
        from perlin_numpy import generate_fractal_noise_2d

        w, h = self.spec["size"]

        rng = np.random.default_rng(seed=self.spec["seed"])
        # our implementation only handles power-of-two dimensions rn
        # https://stackoverflow.com/a/14267557
        def next_power_of_2(x):  
            return 1 if x == 0 else 2**(x - 1).bit_length()
        w_po2 = next_power_of_2(w)
        h_po2 = next_power_of_2(h)
        result = generate_fractal_noise_2d(shape=(w_po2, h_po2),
                                           res=(w_po2 // self.spec["wavelength"], h_po2 // self.spec["wavelength"]),
                                           octaves=self.spec["octaves"],
                                           rng=rng) / np.sqrt(2)
        real_min, real_max = np.amin(result), np.amax(result); print("perlin ->", real_min, real_max)
        result = result[:w, :h]

        # map to (min, max)
        # This is a bit biased in that it always forces full use of the specified range
        min_, max_ = self.spec["range"]
        result = min_ + (max_ - min_) * (result - real_min) / (real_max - real_min)

        # real_min, real_max = np.amin(result), np.amax(result)#; print("adj ->", real_min, real_max)
        # assert real_min >= min_
        # assert real_max <= max_

        np.save(output_paths["$output"], result)

        # Also plot it
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(result.T, origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Perlin noise")
        fig.savefig(output_paths["$output.png"], bbox_inches="tight")

class SrtmExtractOperator(Operator):
    NAME = "srtm_extract"
    VERSION = 1

    def get_inputs(self):
        return set()

    def get_outputs(self):
        return {"$output", "$output.png"}

    def run(self, input_paths, output_paths):
        lat, lon = self.spec["center"]
        size_lon, size_lat = self.spec["size_m"] # longitudinal/latitudinal size *in meters*

        import math
        import matplotlib
        import PIL

        def frac_degrees(degs, mins, secs):
            return degs + mins * (1/60) + secs * (1/60/60)

        lat_center, lon_center = (frac_degrees(*lat), frac_degrees(*lon))

        EARTH_METERS_PER_ARC_SECOND = 30
        DEGREES_PER_METER = frac_degrees(0, 0, 1) / EARTH_METERS_PER_ARC_SECOND

        # N lat, E lon, S lat, W lon
        roi_span = (lat_center + DEGREES_PER_METER * size_lat / 2,
                    lon_center + DEGREES_PER_METER * size_lon / 2,
                    lat_center - DEGREES_PER_METER * size_lat / 2,
                    lon_center - DEGREES_PER_METER * size_lon / 2)
        print(roi_span)

        # Select source heightmap
        # TODO: this needs to be done dynamically
        if True:
            # 3-arc-second dataset (90m)
            # 6000 pixels = 5 deg = 540km
            hmap_filename = Path(__file__).parent / 'inputs/srtm_38_03.tif'
            hmap_meters_per_pixel = 90
            # top, right, bottom, left
            hmap_span = (50.0004170, 10.0004168, 44.9995837, 4.9995835)
        else:
            # 3-arc-second dataset (90m)
            # 6000 pixels = 5 deg = 540km
            hmap_filename = Path(__file__).parent / 'inputs/srtm_39_03.tif'
            hmap_meters_per_pixel = 90
            hmap_span = (50.0004170, 15.0004170, 44.9995837, 9.9995836)

        assert roi_span[0] <= hmap_span[0]
        assert roi_span[1] <= hmap_span[1]
        assert roi_span[2] >= hmap_span[2]
        assert roi_span[3] >= hmap_span[3]

        # Load heightmap
        raw = np.array(PIL.Image.open(hmap_filename))
        hmap = np.copy(raw)
        raw_min = np.amin(hmap)
        clean_min = np.amin(hmap[hmap != -32768])

        hmap[hmap==-32768] = 0

        # Extract & save data
        # SRTM height map is indexed as [row, column], top-to-bottom, left to right
        # TODO: we haven't really thought about sub-sample rounding correctness
        leftmost_pixel = int(math.floor(hmap.shape[1] * (roi_span[3] - hmap_span[3]) / (hmap_span[1] - hmap_span[3])))
        rightmost_pixel = int(math.ceil(hmap.shape[1] * (roi_span[1] - hmap_span[3]) / (hmap_span[1] - hmap_span[3])))
        topmost_pixel = int(math.floor(hmap.shape[0] - hmap.shape[0] * (roi_span[0] - hmap_span[2]) / (hmap_span[0] - hmap_span[2])))
        bottommost_pixel = int(math.ceil(hmap.shape[0] - hmap.shape[0] * (roi_span[2] - hmap_span[2]) / (hmap_span[0] - hmap_span[2])))
        print(f"pixel cut-out: X: {leftmost_pixel}..{rightmost_pixel}, Y: {topmost_pixel}..{bottommost_pixel}")

        with h5py.File(output_paths["$output"], "w") as f:
            HEIGHT_SCALE = 1
            f['heightmap'] = np.flip(hmap[topmost_pixel:bottommost_pixel+1, leftmost_pixel:rightmost_pixel+1].T, axis=1).astype(np.float32) * HEIGHT_SCALE
            f.attrs["heightmap_resolution"] = hmap_meters_per_pixel

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(hmap, origin="upper", extent=(hmap_span[3], hmap_span[1], hmap_span[2], hmap_span[0]), cmap="Greys_r", vmin=-200, vmax=2000)
        ax.set_title(f"Height span: <{clean_min}; {np.amax(hmap)}> m; raw min: {raw_min} m")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        fig.colorbar(im, ax=ax)

        def highlight_roi(top, right, bottom, left):
            rect = matplotlib.patches.Rectangle((left, bottom), right - left, top - bottom, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        highlight_roi(*roi_span)
        fig.savefig(output_paths["$output.png"], bbox_inches="tight", dpi=288)


class TemperatureOperator(Operator):
    NAME = "temperature"
    VERSION = 2

    def get_inputs(self):
        return {"terrain_adj.h5"}

    def get_outputs(self):
        return {"temperature.h5", "temperature.h5.png", "temperature-3d.png"}

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain_adj.h5"], "r") as f:
            heightmap_resolution = f.attrs['heightmap_resolution'][()]
            heightmap = f['heightmap'][()]

        # https://gist.github.com/buzzerrookie/5b6438c603eabf13d07e
        t0 = 288.15
        a = -0.0065
        h0 = 0
        h1 = heightmap
        t1 = t0 + a * (h1 - h0)

        with h5py.File(output_paths["temperature.h5"], "w") as f:
            f['temperature_map'] = t1

        # plot
        extent = tuple(np.array((-0.5, heightmap.shape[0] - 0.5, -0.5, heightmap.shape[1] - 0.5)) * heightmap_resolution)

        fig, ax = plt.subplots()
        im = ax.imshow(t1.T - 273.15, origin='lower', extent=extent, cmap='afmhot')
        ax.set_title("Surface temperature [deg C]")
        ax.set_xlabel("Meters")
        ax.set_ylabel("Meters")
        fig.colorbar(im, ax=ax)
        fig.savefig(output_paths["temperature.h5.png"])

        # experiment 3D plot

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d', elev=60)

        xi = np.arange(0, heightmap.shape[0])
        yi = np.arange(0, heightmap.shape[1])
        X, Y = np.meshgrid(xi, yi)

        # W * H * RGBA
        heatmap = np.zeros((*heightmap.shape, 4))

        from matplotlib import cm

        clim = (np.amin(t1), np.amax(t1))

        for ix, iy in np.ndindex(heightmap.shape):
            heatmap[ix, iy, :] = cm.afmhot((t1[ix, iy] - clim[0]) / (clim[1] - clim[0]))

        ax.plot_surface(X, Y, heightmap.T, facecolors=np.swapaxes(heatmap, 0, 1), rstride=1, cstride=1)
        fig.savefig(output_paths["temperature-3d.png"])


class SunCoverageOperator(Operator):
    NAME = "sun_coverage"
    VERSION = 1

    def get_inputs(self):
        return {"terrain_adj.h5", "water_adj.h5"}

    def get_outputs(self):
        return {#"sun_coverage.h5",
                "sun_coverage.h5.png", "sun_coverage_3d.png"}

    def run(self, input_paths, output_paths):
        with h5py.File(input_paths["terrain_adj.h5"], "r") as f:
            heightmap_resolution = f.attrs['heightmap_resolution'][()]
            heightmap_in = f['heightmap'][()]

        with h5py.File(input_paths["water_adj.h5"], "r") as f:
            surface_level = f["water_surface_level"][()]
            underground_level = f["water_underground_level"][()]
            x_flow = f["water_x_flow"][()]
            y_flow = f["water_y_flow"][()]

        heightmap = surface_level

        extent = tuple(np.array((-0.5, heightmap.shape[0] - 0.5, -0.5, heightmap.shape[1] - 0.5)) * heightmap_resolution)

        sun_coverage = np.zeros_like(heightmap)
        dx = heightmap_resolution / 10   # LOL randum

        # 3 -> 1 pc each (90)
        # 5 -> 2 pc each (45)
        # 7 -> 3 pc each (30)
        # 13 -> 6 pc each (15)
        # 19 -> 9 pc each (10)
        # 37 -> 18 pc each (5)
        # 181 -> 90 pc each (1)

        # simulate for different sun angles. 0 rad represents straight above, positive angles put the sun in 'negative X'
        cnt = 0
        for sun_elevation in np.linspace(-np.pi * 0.5, np.pi * 0.5, 19):
            cnt += 1
            if np.abs(sun_elevation) < 1e-3:
                sun_coverage += 1
                continue

            deg = f"{sun_elevation * 180 / np.pi:.0f}"
            #print(deg)

            result = np.zeros_like(heightmap)

            smooth = 10

            # take into account cosine of the angle, but also shadowing
            # simulate individual rays reaching each target tile
            for x in range(heightmap.shape[0]):
                # the tile is lit if the ray from the reference point and the sun (at infinity) is above terrain at all every tile it passes over
                # generate array of thresholds
                # tan a => height / distance
                #   at a given Y, the heightmap(X', Y) must <= (X - X') * tan(sun elevation) for every X'
                if sun_elevation >= 0:
                    if x == 0:
                        result[x, :] = 1
                        continue

                    # positive, so sun is on the left
                    # -clip; +clip -> 0; 1 == (val + clip) / (2 * clip)
                    xs, ys = np.meshgrid(np.arange(0, x), np.arange(0, heightmap.shape[1]), indexing="ij")
                    threshold = heightmap[x, :] + (x - xs) * dx / np.tan(sun_elevation)
                    is_below_threshold = np.min(np.clip((threshold - heightmap[0:x, :] + smooth) / (2 * smooth), 0, 1), axis=0)

                    result[x, :] = is_below_threshold

                    # if x == heightmap.shape[0] - 1:
                    #     fig, ax = plt.subplots()
                    #     im = ax.imshow(threshold.T, origin='lower', extent=extent, cmap='magma')
                    #     ax.set_title("thresholdo")
                    #     ax.set_xlabel("Meters")
                    #     ax.set_ylabel("Meters")
                    #     fig.colorbar(im, ax=ax)
                    #     fig.savefig(f"TEST1_{deg}.png")
                    #
                    #     fig, ax = plt.subplots()
                    #     im = ax.imshow((heightmap[0:x, :] - threshold).T, origin='lower', extent=extent, cmap='magma')
                    #     ax.set_title("height above threshold (positive -> shadowing)")
                    #     ax.set_xlabel("Meters")
                    #     ax.set_ylabel("Meters")
                    #     fig.colorbar(im, ax=ax)
                    #     fig.savefig(f"TEST2_{deg}.png")
                    #
                    #     fig, ax = plt.subplots()
                    #     im = ax.imshow(result.T, origin='lower', extent=extent, cmap='magma')
                    #     ax.set_title("result")
                    #     ax.set_xlabel("Meters")
                    #     ax.set_ylabel("Meters")
                    #     fig.colorbar(im, ax=ax)
                    #     fig.savefig(f"TEST3_{deg}.png")
                else:
                    if x == heightmap.shape[0] - 1:
                        result[x, :] = 1

                        # fig, ax = plt.subplots()
                        # im = ax.imshow(result.T, origin='lower', extent=extent, cmap='magma')
                        # ax.set_title("result")
                        # ax.set_xlabel("Meters")
                        # ax.set_ylabel("Meters")
                        # fig.colorbar(im, ax=ax)
                        # fig.savefig(f"TEST3_{deg}.png")
                        continue

                    # negative, so sun is on the right
                    xs, ys = np.meshgrid(np.arange(x + 1, heightmap.shape[0]), np.arange(0, heightmap.shape[1]), indexing="ij")
                    threshold = heightmap[x, :] + (x - xs) * dx / np.tan(sun_elevation)
                    is_below_threshold = np.min(np.clip((threshold - heightmap[x + 1:, :] + smooth) / (2 * smooth), 0, 1), axis=0)

                    result[x, :] = is_below_threshold

                    # if x == 20:
                    #     fig, ax = plt.subplots()
                    #     im = ax.imshow(threshold.T, origin='lower', extent=extent, cmap='magma')
                    #     ax.set_title("thresholdo")
                    #     ax.set_xlabel("Meters")
                    #     ax.set_ylabel("Meters")
                    #     fig.colorbar(im, ax=ax)
                    #     fig.savefig(f"TEST1_{deg}.png")
                    #
                    #     fig, ax = plt.subplots()
                    #     im = ax.imshow(np.clip(heightmap[x + 1:, :] - threshold, -10, 10).T, origin='lower', extent=extent, cmap='magma')
                    #     ax.set_title("height above threshold (positive -> shadowing)")
                    #     ax.set_xlabel("Meters")
                    #     ax.set_ylabel("Meters")
                    #     fig.colorbar(im, ax=ax)
                    #     fig.savefig(f"TEST2_{deg}.png")

            sun_coverage += np.cos(sun_elevation) * result

        sun_coverage /= cnt

        fig, ax = plt.subplots()
        im = ax.imshow(sun_coverage.T, vmin=0, origin='lower', extent=extent, cmap='pink')
        ax.set_title("Cumulative sun coverage")
        ax.set_xlabel("Meters")
        ax.set_ylabel("Meters")
        fig.colorbar(im, ax=ax)
        fig.savefig(output_paths["sun_coverage.h5.png"])

        # experiment 3D plot

        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111, projection='3d', elev=60)

        xi = np.arange(0, heightmap.shape[0])
        yi = np.arange(0, heightmap.shape[1])
        X, Y = np.meshgrid(xi, yi)

        # W * H * RGBA
        heatmap = np.zeros((*heightmap.shape, 4))

        from matplotlib import cm

        clim = (np.amin(sun_coverage), np.amax(sun_coverage))

        for ix, iy in np.ndindex(heightmap.shape):
            heatmap[ix, iy, :] = cm.pink((sun_coverage[ix, iy] - clim[0]) / (clim[1] - clim[0]))

        ax.plot_surface(X, Y, heightmap.T, facecolors=np.swapaxes(heatmap, 0, 1), rstride=1, cstride=1)
        fig.savefig(output_paths["sun_coverage_3d.png"])

        # with h5py.File(output_paths["sun_coverage.h5"], "w") as f:
        #     f['sun_coverage_map'] = sun_coverage

pipeline_world = Pipeline(all_steps=[
        BlenderRenderStep,
        GeoStep,
        GeoPlotStep,
        HydroAdjustStep,
        HydroPlotsStep,
        HydroSimStep,
        PerlinOperator,
        PlyExportStep,
        SrtmExtractOperator,
        SunCoverageOperator,
        TemperatureOperator,
    ])
