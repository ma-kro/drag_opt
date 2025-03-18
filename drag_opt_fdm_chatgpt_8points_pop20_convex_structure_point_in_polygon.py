import numpy as np
import matplotlib.pyplot as plt
import os
from deap import base, creator, tools, algorithms

# -----------------------------------------------------------------------------
# 0) Hilfsfunktionen: Konvexe Hülle + Ray Casting
# -----------------------------------------------------------------------------
def convex_hull(points):
    """
    Berechnet die konvexe Hülle (Monotone chain) einer 2D-Punktwolke.
    Gibt eine Liste von Eckpunkten in CCW zurück.
    """
    pts = sorted(points)
    unique_pts = []
    for p in pts:
        if not unique_pts or (unique_pts[-1] != p):
            unique_pts.append(p)
    pts = unique_pts
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    lower.pop()
    upper.pop()
    return lower + upper

def is_point_in_polygon(px, py, polygon):
    """
    Ray-Casting-Test: True, wenn (px,py) innerhalb des Polygons liegt.
    Erwartet polygon als Liste [(x1,y1), (x2,y2), ...] in CCW oder CW.
    """
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        # Prüfe, ob der horizontale Strahl durch die Kante (x1,y1)-(x2,y2) geht
        intersects = ((y1 > py) != (y2 > py))  # => Der Strahl auf Höhe py kreuzt?
        if intersects:
            # Wo schneidet die Kante die horizontale Linie y=py?
            # x-Koordinate an der Schnittstelle:
            x_intersect = x1 + (py - y1)*(x2 - x1)/(y2 - y1)
            if x_intersect > px:
                inside = not inside
    return inside

# -----------------------------------------------------------------------------
# 1) Evolutionärer Algorithmus (DEAP) - Setup
# -----------------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual(num_points=10):
    """
    Erzeugt eine Menge Basispunkte, die wir später zu einer konvexen Hülle formen.
    """
    base_pts = []
    for _ in range(num_points):
        x_coord = np.random.uniform(0.8, 2.2)
        y_coord = np.random.uniform(-0.4, 0.4)
        base_pts.append((x_coord, y_coord))
    return base_pts

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------------------------------------------------------
# 2) Navier-Stokes-Simulation (Punkt-in-Polygon)
# -----------------------------------------------------------------------------
def simulate_navier_stokes(base_points,
                           nx=60, ny=40,
                           Re=50.0,
                           U_in=1.0,
                           max_iter=3000,
                           tol=1e-5):
    """
    1) Konvexe Hülle der Basispunkte => polygon.
    2) obstacle_mask = 'inside polygon' via Ray-Casting.
    3) FDM-Navier-Stokes.
    """
    # 2.1) Konvexes Polygon
    polygon = convex_hull(base_points)
    # Falls zuwenig Punkte => triviale Dopplung
    if len(polygon) < 3:
        polygon = polygon * 2

    # Diskretisierungs-Parameter
    x_min, x_max = 0.0, 3.0
    y_min, y_max = -0.5, 0.5
    dx = (x_max - x_min)/(nx-1)
    dy = (y_max - y_min)/(ny-1)
    nu = U_in/Re

    # Gitter
    x_arr = np.linspace(x_min, x_max, nx)
    y_arr = np.linspace(y_min, y_max, ny)
    Xgrid, Ygrid = np.meshgrid(x_arr, y_arr)  # shape (ny, nx)

    # Felder
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Einströmprofil
    u[:, 0] = U_in

    # 2.2) obstacle_mask: Ray-Casting
    obstacle_mask = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            xx = Xgrid[i, j]
            yy = Ygrid[i, j]
            # Im Polygon?
            if is_point_in_polygon(xx, yy, polygon):
                obstacle_mask[i, j] = True

    def apply_boundary_conditions(u_, v_, p_):
        # Einlass (x=0)
        u_[:, 0] = U_in
        v_[:, 0] = 0.0
        # Auslass (x=3) => du/dx=0, dv/dx=0, dp/dx=0
        u_[:, -1] = u_[:, -2]
        v_[:, -1] = v_[:, -2]
        p_[:, -1] = p_[:, -2]
        # oben/unten => No-Slip
        u_[0, :] = 0.0
        v_[0, :] = 0.0
        u_[-1, :] = 0.0
        v_[-1, :] = 0.0
        # Hindernis => No-Slip
        u_[obstacle_mask] = 0.0
        v_[obstacle_mask] = 0.0

    # Zeitschritt
    dt = 0.05 * min(dx, dy)/U_in

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        u_old = u.copy()
        v_old = v.copy()

        # Vorhersage (ue, ve)
        ue = u.copy()
        ve = v.copy()

        lap_u = np.zeros_like(u)
        lap_v = np.zeros_like(u)
        dudx  = np.zeros_like(u)
        dudy  = np.zeros_like(u)
        dvdx  = np.zeros_like(u)
        dvdy  = np.zeros_like(u)

        # Laplace + zentrale Differenzen => innen
        lap_u[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2 +
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
        )
        lap_v[1:-1, 1:-1] = (
            (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2])/dx**2 +
            (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1])/dy**2
        )

        dudx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2])/(2*dx)
        dudy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)
        dvdx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dx)
        dvdy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy)

        # ue, ve
        ue[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            - dt*(u[1:-1, 1:-1]*dudx[1:-1, 1:-1] +
                  v[1:-1, 1:-1]*dudy[1:-1, 1:-1])
            + nu*dt*lap_u[1:-1, 1:-1]
        )
        ve[1:-1, 1:-1] = (
            v[1:-1, 1:-1]
            - dt*(u[1:-1, 1:-1]*dvdx[1:-1, 1:-1] +
                  v[1:-1, 1:-1]*dvdy[1:-1, 1:-1])
            + nu*dt*lap_v[1:-1, 1:-1]
        )

        # NaN/Inf-Check
        if (not np.isfinite(ue).all()) or (not np.isfinite(ve).all()):
            return 1e6, u, v, p

        # Druck-Poisson
        p_old = p.copy()
        for _ in range(50):
            p[1:-1, 1:-1] = (
                (p_old[1:-1, 2:] + p_old[1:-1, :-2] +
                 p_old[2:, 1:-1] + p_old[:-2, 1:-1])/4.0
                - (dx*dy)/(4.0*dt)*(
                    (ue[1:-1, 2:] - ue[1:-1, :-2])/(2*dx) +
                    (ve[2:, 1:-1] - ve[:-2, 1:-1])/(2*dy)
                )
            )
            p[:, -1] = p[:, -2]
            p[:, 0]  = p[:, 1]
            p[0, :]  = p[1, :]
            p[-1, :] = p[-2, :]
            p_old = p.copy()
            if not np.isfinite(p).all():
                return 1e6, u, v, p

        # Geschwindigkeitskorrektur
        u[1:-1, 1:-1] = ue[1:-1, 1:-1] - dt*(p[1:-1, 2:] - p[1:-1, :-2])/(2*dx)
        v[1:-1, 1:-1] = ve[1:-1, 1:-1] - dt*(p[2:, 1:-1] - p[:-2, 1:-1])/(2*dy)

        # Randbedingungen
        apply_boundary_conditions(u, v, p)

        if (not np.isfinite(u).all()) or (not np.isfinite(v).all()):
            return 1e6, u, v, p

        # Konvergenz
        diff_u = np.linalg.norm(u - u_old, ord=2)
        diff_v = np.linalg.norm(v - v_old, ord=2)
        if diff_u < tol and diff_v < tol:
            break

    # CW
    fy = (y_max - y_min)
    fx = np.sum(p[:, 0] - p[:, -1])*(fy/ny)

    # Effektive "Querfläche" => wir nehmen bounding box. 
    # Oder Du definierst was Du willst. Hier: bounding box des Polygons.
    poly_x = [pt[0] for pt in polygon]
    poly_y = [pt[1] for pt in polygon]
    if len(poly_x)>0:
        obs_xmin, obs_xmax = min(poly_x), max(poly_x)
        obs_ymin, obs_ymax = min(poly_y), max(poly_y)
        cross_area = (obs_ymax - obs_ymin)
    else:
        cross_area = 1e-10

    if cross_area < 1e-10:
        cross_area = 1e-10

    cw = fx/(0.5*(U_in**2)*cross_area)
    return abs(cw), u, v, p

def calculate_cw_ns(base_points, plot_dir=None):
    """
    Führt die NS-Simulation mit Ray-Casting (echte Hindernisform) durch.
    """
    cw, u, v, p = simulate_navier_stokes(base_points)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        x_min, x_max = 0.0, 3.0
        y_min, y_max = -0.5, 0.5
        nx_ = u.shape[1]
        ny_ = u.shape[0]
        xg = np.linspace(x_min, x_max, nx_)
        yg = np.linspace(y_min, y_max, ny_)
        X, Y = np.meshgrid(xg, yg)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.contourf(X, Y, p, levels=20)
        plt.colorbar(label='Druck')
        plt.title(f"Druckfeld (CW={cw:.2f})")

        speed = np.sqrt(u**2 + v**2)
        plt.subplot(1,2,2)
        plt.streamplot(X, Y, u, v, density=1.2, color='k')
        plt.contourf(X, Y, speed, alpha=0.5)
        plt.colorbar(label='|u|')
        plt.title("Geschwindigkeitsfeld")

        plt.savefig(os.path.join(plot_dir, "ns_solution.png"))
        plt.close()

    return cw

# -----------------------------------------------------------------------------
# 3) Genetische Operatoren
# -----------------------------------------------------------------------------
def mutate_base_points(individual, indpb=0.3):
    """
    Verschiebt die Basispunkte, wodurch sich das Polygon (konvexe Hülle) ändert.
    """
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            x_shift = np.random.uniform(-0.5, 0.5)
            y_shift = np.random.uniform(-0.25, 0.25)
            individual[i] = (individual[i][0] + x_shift,
                             individual[i][1] + y_shift)
    return (individual,)

def crossover_base_points(ind1, ind2, indpb=0.7):
    """
    Tauscht einzelne Basispunkte zweier Individuen.
    """
    for i in range(len(ind1)):
        if np.random.rand() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

toolbox.register("evaluate", lambda base_pts: (calculate_cw_ns(base_pts),))
toolbox.register("mate", crossover_base_points)
toolbox.register("mutate", mutate_base_points, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------------------------------------------------------
# 4) Visualisierung & Speicherung
# -----------------------------------------------------------------------------
def save_generation(population, gen):
    """
    Erzeugt aus den Basis-Punkten die konvexe Hülle, speichert Polygon-Plot + Strömungsfeld.
    """
    from math import isnan

    if isinstance(gen, int):
        gen_dir = f"results/gen_{gen:02d}"
    else:
        gen_dir = f"results/{gen}"
    os.makedirs(gen_dir, exist_ok=True)
    
    for idx, base_pts in enumerate(population):
        ind_dir = os.path.join(gen_dir, f"ind_{idx:02d}")
        os.makedirs(ind_dir, exist_ok=True)

        # Konvexe Hülle
        hull = convex_hull(base_pts)
        if len(hull) < 2:
            hull = hull * 2
        cw_val = base_pts.fitness.values[0]

        # Plot
        px, py = zip(*(hull + [hull[0]]))
        plt.figure(figsize=(6,4))
        plt.plot(px, py, '-o')
        plt.fill(px, py, alpha=0.3)
        plt.xlim(0, 3)
        plt.ylim(-0.6, 0.6)
        plt.title(f"Struktur (CW: {cw_val:.2f})")
        plt.savefig(os.path.join(ind_dir, "form.png"))
        plt.close()

        # Strömungsfeld
        calculate_cw_ns(base_pts, plot_dir=ind_dir)

def plot_progress(logbook):
    gens = logbook.select("gen")
    avg_vals = logbook.select("avg")
    best_vals = logbook.select("min")

    plt.figure(figsize=(8,5))
    plt.plot(gens, avg_vals, label="Durchschnitt")
    plt.plot(gens, best_vals, '--', label="Bester")
    plt.xlabel("Generation")
    plt.ylabel("CW-Wert")
    plt.title("Optimierungsverlauf")
    plt.grid(True)
    plt.legend()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/optimization_progress.png")
    plt.close()

# -----------------------------------------------------------------------------
# 5) Hauptprogramm
# -----------------------------------------------------------------------------
def main():
    pop_size = 30
    n_generations = 30

    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    logbook = tools.Logbook()

    for gen in range(n_generations):
        # Fitnessberechnung
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Visualisierung
        save_generation(population, gen)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # Evolutionärer Schritt
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.7, mutpb=0.4)
        population[:] = offspring

    # Abschluss
    plot_progress(logbook)

    # Bestes Individuum
    best_ind = tools.selBest(population, 1)[0]
    save_generation([best_ind], "best")

if __name__ == "__main__":
    main()
