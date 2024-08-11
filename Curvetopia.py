import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from svgpathtools import svg2paths
import svgwrite

def parse_svg(svg_path):
    paths, attributes = svg2paths(svg_path)
    path_data = []
    for path in paths:
        points = []
        for segment in path:
            if segment.start != segment.end:
                points.append([segment.start.real, segment.start.imag])
                points.append([segment.end.real, segment.end.imag])
        if points:
            path_data.append(np.array(points))
    return path_data

def to_svg(paths, output_svg):
    dwg = svgwrite.Drawing(output_svg, profile='tiny')
    for path in paths:
        d = 'M ' + ' '.join(f'{x},{y}' for x, y in path)
        dwg.add(dwg.path(d=d, stroke='black', fill='none'))
    dwg.save()

def fit_straight_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    return np.column_stack((X.flatten(), y_pred))

def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x*2 + y*2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc*2 + yc*2)
    return xc, yc, r

def fit_quadratic_bezier(points):
    n = len(points)
    P0 = points[0]
    P2 = points[-1]
    P1 = np.mean(points[1:-1], axis=0)
    t = np.linspace(0, 1, n)
    bezier_points = (1 - t)[:, np.newaxis] ** 2 * P0 + 2 * (1 - t)[:, np.newaxis] * t[:, np.newaxis] * P1 + t[:, np.newaxis] ** 2 * P2
    return bezier_points

def is_rectangle(points):
    if len(points) != 4:
        return False
    vectors = [points[i] - points[i-1] for i in range(4)]
    norms = [np.linalg.norm(v) for v in vectors]
    if any(norm == 0 for norm in norms):
        return False
    angles = []
    for i in range(4):
        norm_product = norms[i] * norms[(i-1) % 4]
        if norm_product == 0:
            return False
        angle = np.arccos(np.dot(vectors[i], vectors[(i-1) % 4]) / norm_product)
        angles.append(angle)
    return np.allclose(angles, [np.pi/2]*4)

def is_star_shape(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_angles = np.sort(angles)
    differences = np.diff(sorted_angles)
    return np.allclose(differences, np.mean(differences))

def detect_reflection_symmetry(points):
    center = np.mean(points, axis=0)
    points_centered = points - center
    distances = np.linalg.norm(points_centered, axis=1)
    return np.allclose(distances, distances[::-1])

def complete_curve(points):
    if len(points) < 4:
        return points
    tck, u = splprep([points[:, 0], points[:, 1]], s=0)
    unew = np.linspace(0, 1, len(points)*2)
    out = splev(unew, tck)
    completed_curve = np.column_stack(out)
    return completed_curve

def regularize_shape(points):
    if len(points) < 4:
        return points
    if is_rectangle(points):
        return points
    if is_star_shape(points):
        return points
    xc, yc, r = fit_circle(points)
    circle_fit = np.column_stack((xc + r * np.cos(np.linspace(0, 2 * np.pi, len(points))),
                                  yc + r * np.sin(np.linspace(0, 2 * np.pi, len(points)))))
    return circle_fit

def smooth_curve(points, sigma=1):
    if len(points) < 2:
        return points
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    return np.column_stack((smoothed_x, smoothed_y))

def process_data(paths):
    processed_paths = []
    for path in paths:
        if len(path) < 2:
            continue
        regularized_path = regularize_shape(path)
        smoothed_path = smooth_curve(regularized_path)
        bezier_path = fit_quadratic_bezier(smoothed_path)
        if detect_reflection_symmetry(bezier_path):
            completed_path = complete_curve(bezier_path)
        else:
            completed_path = bezier_path
        processed_paths.append(completed_path)
    return processed_paths

def plot(paths, title, ax):
    colours = ['red', 'green', 'blue', 'yellow', 'purple']
    for i, path in enumerate(paths):
        c = colours[i % len(colours)]
        ax.plot(path[:, 0], path[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

def main():
    input_svg = 'frag0.svg'
    output_svg = 'output.svg'
    paths = parse_svg(input_svg)
    processed_paths = process_data(paths)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot(processed_paths, 'Processed Data', ax)
    plt.savefig(output_svg)
    plt.show()

if __name__ == '__main__':
    main()
