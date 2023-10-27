from dataclasses import dataclass
import numpy as np


@dataclass
class Init:
    surface_level: np.ndarray
    x_flow: np.ndarray
    y_flow: np.ndarray


@dataclass
class Result:
    heightmap: np.ndarray
    surface_level: np.ndarray
    x_flow: np.ndarray
    y_flow: np.ndarray

    surface_level_changelog: np.ndarray
    x_flow_changelog: np.ndarray
    y_flow_changelog: np.ndarray
    volume_changelog: np.ndarray
    dt_changelog: np.ndarray


# Coordinate system: axis 0 = X (left -> right), axis 1 = Y (bottom -> top)
def simulate(*, heightmap,
             L_boundary_top=None, L_boundary_bot=None, L_boundary_left=None, L_boundary_right=None,
             viscosity, rainfall,
             max_level_change, max_flow_change, max_iters,
             erosion_strength=None, erosion_sideways_coeff=None,
             init=None, verbose=False):
    heightmap = np.copy(heightmap)
    cols, rows = heightmap.shape

    if init is None:
        levels = np.copy(heightmap)
        x_flow = np.zeros((cols + 1, rows))
        y_flow = np.zeros((cols, rows + 1))
    else:
        levels = init.surface_level
        x_flow = init.x_flow
        y_flow = init.y_flow

    if L_boundary_top is None: L_boundary_top = np.zeros(cols)
    if L_boundary_bot is None: L_boundary_bot = np.zeros(cols)
    if L_boundary_left is None: L_boundary_left = np.zeros(rows)
    if L_boundary_right is None: L_boundary_right = np.zeros(rows)

    x_flow_new = np.zeros((cols + 1, rows))
    y_flow_new = np.zeros((cols, rows + 1))

    surface_level_changelog = []
    x_flow_changelog = []
    y_flow_changelog = []
    volume_changelog = []
    dt_changelog = []

    """
    Convergence strategy:
    - begin with a very conservative dt, like 0.1 (and why this value should good ?)
    - set limits: min 0.001, max 1
    - when converging well (surface_level change decrease of at least 20% in the last 100 simulated time units)
        - multiply dt by 2
    - when not converging well (surface_level change decrease of at less than 10% in the last 100 simulated time units)
        - divide dt by 2

    Drawback: we should apply different logic in the beginning when lakes are filling up -> there the level change will be constant for a long time
    """

    min_dt = 0.0005
    max_dt = 1
    dt = 0.2
    alpha = 1

    for iter_ in range(1, max_iters + 1):
        # Up to 4 neighbors can steal water from any given cell.
        # Make sure the remaining quantity cannot go negative in a single simulation step.
        # (Not like this really works, since the flow is only updated by a factor of alpha every iteration)
        # one_over_4dt = 1 / 4 / dt
        one_over_4dt = 1

        # Horzflow

        # Left boundary Hflow
        x_flow_new[0, :] = (L_boundary_left - levels[0, :]) / viscosity
        x_flow_new[0, :] = np.maximum(x_flow_new[0, :], -np.maximum(0, levels[0, :] - heightmap[0, :]) * one_over_4dt)

        # Right boundary Hflow
        x_flow_new[cols, :] = (levels[cols - 1, :] - L_boundary_right) / viscosity
        x_flow_new[cols, :] = np.minimum(x_flow_new[cols, :], np.maximum(0, levels[cols - 1, :] - heightmap[cols - 1, :]) * one_over_4dt)

        # Bulk Hflow
        level_left = levels[0:cols - 1, :]
        level_right = levels[1:cols, :]
        height_left = heightmap[0:cols - 1, :]
        height_right = heightmap[1:cols, :]
        x_flow_new[1:cols, :] = (level_left - level_right) / viscosity

        x_flow_new[1:cols, :] = np.minimum(x_flow_new[1:cols, :], np.maximum(0, level_left - height_left) * one_over_4dt)
        x_flow_new[1:cols, :] = np.maximum(x_flow_new[1:cols, :], -np.maximum(0, level_right - height_right) * one_over_4dt)

        x_flow_change = np.amax(np.abs(x_flow_new - x_flow))
        x_flow += (x_flow_new - x_flow) * dt * alpha

        # Vertflow

        # South boundary Hflow
        y_flow_new[:, 0] = (L_boundary_bot - levels[:, 0]) / viscosity
        y_flow_new[:, 0] = np.maximum(y_flow_new[:, 0], -np.maximum(0, levels[:, 0] - heightmap[:, 0]) * one_over_4dt)

        # North boundary Hflow
        y_flow_new[:, rows] = (levels[:, rows - 1] - L_boundary_top) / viscosity
        y_flow_new[:, rows] = np.minimum(y_flow_new[:, rows], np.maximum(0, levels[:, rows - 1] - heightmap[:, rows - 1]) * one_over_4dt)

        # Bulk Vflow
        level_south = levels[:, 0:rows - 1]
        level_north = levels[:, 1:rows]
        height_south = heightmap[:, 0:rows - 1]
        height_north = heightmap[:, 1:rows]
        y_flow_new[:, 1:rows] = (level_south - level_north) / viscosity

        y_flow_new[:, 1:rows] = np.minimum(y_flow_new[:, 1:rows], np.maximum(0, level_south - height_south) * one_over_4dt)
        y_flow_new[:, 1:rows] = np.maximum(y_flow_new[:, 1:rows], -np.maximum(0, level_north - height_north) * one_over_4dt)

        y_flow_change = np.amax(np.abs(y_flow_new - y_flow))
        y_flow += (y_flow_new - y_flow) * dt * alpha

        # Update levels

        dlevel = x_flow[0:cols, 0:rows] - x_flow[1:cols+1, 0:rows] \
               + y_flow[0:cols, 0:rows] - y_flow[0:cols, 1:rows+1] \
               + rainfall
        levels += dlevel * dt
        level_change = np.amax(np.abs(dlevel))

        # Erosion
        if erosion_strength is not None:
            speed_x = (x_flow[0:cols, :] + x_flow[1:cols + 1, :]) / 2
            speed_y = (y_flow[:, 0:rows] + y_flow[:, 1:rows + 1]) / 2
            speed = np.sqrt(np.power(speed_x, 2) + np.power(speed_y, 2))

            # Spread to neighbors
            #  the logic is own_speed = max(own_speed, max(neighbor_speeds) * sideways_coeff)
            left_speed = speed[0:cols-2,1:rows-1]
            right_speed = speed[2:cols,1:rows-1]
            south_speed = speed[1:cols-1,0:rows-2]
            north_speed = speed[1:cols-1,2:rows]
            max_neighbor_speed = np.maximum(np.maximum(left_speed, right_speed), np.maximum(north_speed, south_speed))
            speed[1:cols-1, 1:rows-1] = np.maximum(speed[1:cols-1, 1:rows-1], max_neighbor_speed * erosion_sideways_coeff)

            heightmap += -speed * erosion_strength

        total_volume = np.sum(np.maximum(0, levels - heightmap))

        surface_level_changelog.append(level_change)
        x_flow_changelog.append(x_flow_change)
        y_flow_changelog.append(y_flow_change)
        volume_changelog.append(total_volume)
        dt_changelog.append(dt)

        flow_change_in_limit = (level_change < max_level_change and x_flow_change < max_flow_change and y_flow_change < max_flow_change)
        if iter_ % 1000 == 0 or flow_change_in_limit:
            if verbose:
                print(f"Iteration {iter_:6d}: {dt=:.3f} {level_change=:6.3f} {x_flow_change=:.5f} {y_flow_change=:.5f} max_level={np.amax(levels):.2f} {total_volume=:.1f}")

            if flow_change_in_limit:
                break

        INTERVAL = 1000

        if erosion_strength is None and iter_ % (INTERVAL * 2) == 0:
            # The sampling of INT points is only done every 2*INT steps, so that after every adjustment there is some space to stabilize
            if len(surface_level_changelog) > INTERVAL:
                dt_interval = INTERVAL * dt
                volume_increase_rate = np.log(volume_changelog[-1] / volume_changelog[-1 - INTERVAL]) / dt_interval

                if volume_increase_rate < np.log(1.03) / 100:
                    volume_stable = True
                else:
                    print(f"\tVolume unstable: log rate {volume_increase_rate=:.3g} vs {np.log(1.03) / 100:.5f}")
                    volume_stable = False

                # Estimate the `alpha` for surface_level_changelog(t) = exp(alpha * t)
                lin, const = np.polyfit(np.linspace(0, dt_interval, INTERVAL), np.log(surface_level_changelog[-INTERVAL:]), 1)
                growth = lin
                print(f"\tApproximate recent trend of dh/dt: {np.exp(growth * 200):.3g}x over 200 time-steps")

                if volume_stable:
                    if growth < np.log(0.25) / 200 and dt * 2 <= max_dt:
                        dt *= 1.25
                        print(f"\tGood convergence (log rate {growth=:.3g} vs {np.log(0.25) / 200:.3g}), go faster")
                    elif growth > np.log(0.8) / 200 and dt / 2 >= min_dt:
                        dt /= 1.25
                        print(f"\tUnsatisfactory convergence (log rate {growth=:.3g} vs {np.log(0.8) / 200:.3g}), let's slow down")
                else:
                    # TODO: Detect & treat gross instability?
                    pass

    return Result(heightmap=heightmap,
                  surface_level=levels,
                  x_flow=x_flow,
                  y_flow=y_flow,
                  surface_level_changelog=np.array(surface_level_changelog),
                  x_flow_changelog=np.array(x_flow_changelog),
                  y_flow_changelog=np.array(y_flow_changelog),
                  volume_changelog=np.array(volume_changelog),
                  dt_changelog=np.array(dt_changelog),
                  )


def adjust_underground_levels(surface_level, heightmap, threshold):
    levels = surface_level
    base_h = heightmap
    M, N = surface_level.shape

    # Spread water under terrain
    levels_adj = np.copy(levels)
    below_terrain = np.less(levels - base_h, threshold)
    levels_adj[below_terrain] = np.nan

    while np.isnan(levels_adj).any():
        def try_move(to, from_):
            dest_nan = np.isnan(levels_adj[to])
            src_not_nan = np.logical_not(np.isnan(levels_adj[from_]))
            dest_terrain_below_src_level = np.less(levels_adj[from_] - base_h[to], threshold)

            mask = np.logical_and(np.logical_and(dest_nan, src_not_nan), dest_terrain_below_src_level)

            np.copyto(levels_adj[to], levels_adj[from_], where=mask)
            return np.any(mask)

        # right->left
        a = try_move(np.s_[:,0:N-1], np.s_[:,1:N])
        # left -> right
        b = try_move(np.s_[:,1:N], np.s_[:,0:N-1])
        # north -> south
        c = try_move(np.s_[1:M,:], np.s_[0:M-1,:])
        # south -> north
        d = try_move(np.s_[0:M-1,:], np.s_[1:M,:])

        if not any((a, b, c, d)):
            break

    # If there are still any gaps left, set them to THRESHOLD below ground surface
    levels_adj[np.isnan(levels_adj)] = base_h[np.isnan(levels_adj)] - threshold

    return levels_adj
