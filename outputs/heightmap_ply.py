from plyfile import PlyData, PlyElement
import matplotlib
import numpy as np


def tet_ply(heightmap, water_column, h_scale, heightmap_resolution, text, byte_order):
    #xs, ys = np.meshgrid(np.arange(heightmap.shape[0]), np.arange(heightmap.shape[1]))

    scale = 1 / max(heightmap.shape[0] * heightmap_resolution,
                    heightmap.shape[1] * heightmap_resolution)
    h_scale *= scale

    sm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, 1),
                                      matplotlib.pyplot.get_cmap('gist_earth'))

    vertex_data = []

    for y, x in np.ndindex(heightmap.shape[::-1]):
        h = heightmap[x, y]
        rgba = sm.to_rgba(h / 2000)

        vertex_data += [(x * heightmap_resolution * scale,
                         y * heightmap_resolution * scale,
                         h * h_scale,
                         int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)) ]

    offset1 = len(vertex_data)
    for y, x in np.ndindex(heightmap.shape[::-1]):
        h = heightmap[x, y] + water_column[x, y]

        vertex_data += [(x * heightmap_resolution * scale,
                         y * heightmap_resolution * scale,
                         h * h_scale,
                         0x33, 0x88, 0xCC) ]

    vertex = np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    face_data = []

    for x, y in np.ndindex((heightmap.shape[0]-1, heightmap.shape[1]-1)):
        lt = y * heightmap.shape[0] + x
        rt = y * heightmap.shape[0] + (x+1)
        lb = (y+1) * heightmap.shape[0] + x
        rb = (y+1) * heightmap.shape[0] + (x+1)

        if (x + y) % 2 == 0:
            face_data += [([lb, rt, lt],), ([lb, rb, rt],)]
        else:
            face_data += [([lt, lb, rb],), ([lt, rb, rt],)]

    for x, y in np.ndindex((heightmap.shape[0]-1, heightmap.shape[1]-1)):
        lt = offset1+y * heightmap.shape[0] + x
        rt = offset1+y * heightmap.shape[0] + (x+1)
        lb = offset1+(y+1) * heightmap.shape[0] + x
        rb = offset1+(y+1) * heightmap.shape[0] + (x+1)

        if (x + y) % 2 == 0:
            face_data += [([lb, rt, lt],), ([lb, rb, rt],)]
        else:
            face_data += [([lt, lb, rb],), ([lt, rb, rt],)]

    face = np.array(face_data, dtype=[('vertex_indices', 'i4', (3,))])

    return PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['tetrahedron vertices']
            ),
            PlyElement.describe(face, 'face')
        ],
        text=text, byte_order=byte_order,
        comments=['single tetrahedron with colored faces']
    )
