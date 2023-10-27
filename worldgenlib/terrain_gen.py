import numpy as np

# Implements https://en.wikipedia.org/wiki/Diamond-square_algorithm
def random_midpoint_displacement_fractal(rng, num_squares, square_size, base_level, primary_scale, roughness):
    h = np.zeros((num_squares[0] * square_size + 1, num_squares[1] * square_size + 1))
    #touched = np.zeros(h.shape, dtype=np.int)

    # Cast primary_scale & roughness to 2D arrays
    if not isinstance(primary_scale, np.ndarray):
        primary_scale = np.full_like(h, primary_scale)
    else:
        assert primary_scale.shape == h.shape

    if not isinstance(roughness, np.ndarray):
        roughness = np.full_like(h, roughness)
    else:
        assert roughness.shape == h.shape

    # start with the corners
    for y in range(num_squares[1] + 1):
        for x in range(num_squares[0] + 1):
            h[x * square_size, y * square_size] = base_level + rng.exponential(primary_scale[x * square_size, y * square_size])

#     plt.figure()
#     plt.imshow(h, cmap='gray')
#     plt.colorbar()

    while square_size >= 2:
        assert square_size % 2 == 0

        #print('ITER', iteration, 'squares', num_squares, 'square_size', square_size)

        # "diamond" step
        # randoms = rng.normal(loc=0.0, scale=square_size * roughness, size=num_squares)

        for y in range(num_squares[1]):
            for x in range(num_squares[0]):
                # sample 4 corners
                c1 = h[x * square_size, y * square_size]
                c2 = h[x * square_size, (y + 1) * square_size]
                c3 = h[(x + 1) * square_size, (y + 1) * square_size]
                c4 = h[(x + 1) * square_size, y * square_size]
                c = np.mean([c1, c2, c3, c4])

                displacement = rng.normal(loc=0.0, scale=square_size * roughness[x * square_size + square_size // 2, y * square_size + square_size // 2])
                h[x * square_size + square_size // 2, y * square_size + square_size // 2] = c + displacement
                #touched[x * square_size + square_size // 2, y * square_size + square_size // 2] += 1

        num_squares = (num_squares[0] * 2, num_squares[1] * 2)
        square_size //= 2

        # "square" step
        for y in range(0, num_squares[1] + 1):
            if y % 2 == 0:
                xs = range(1, num_squares[0], 2)
            else:
                xs = range(0, num_squares[0] + 1, 2)

            # TODO: this is needlessly wasteful
            # randoms = rng.normal(loc=0.0, scale=square_size * roughness, size=(num_squares[0] + 1))

            for x in xs:
                # sample 4 directions
                nan = float('NaN')
                c1 = h[(x - 1) * square_size, y * square_size] if x > 0 else nan
                c2 = h[x * square_size, (y - 1) * square_size] if y > 0 else nan
                c3 = h[(x + 1) * square_size, y * square_size] if x < num_squares[0]-1 else nan
                c4 = h[x * square_size, (y + 1) * square_size] if y < num_squares[1]-1 else nan
                c = np.nanmean([c1, c2, c3, c4])

                displacement = rng.normal(loc=0.0, scale=square_size * roughness[x * square_size, y * square_size])
                h[x * square_size, y * square_size] = c + displacement
                #touched[x * square_size, y * square_size] += 1

#         plt.figure()
#         plt.imshow(h, cmap='gray')
#         plt.colorbar()

#     plt.figure()
#     plt.imshow(touched)
#     plt.colorbar()
    return h
