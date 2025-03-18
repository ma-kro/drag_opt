import numpy as np
import matplotlib.pyplot as plt
import os
from deap import base, creator, tools, algorithms

# -----------------------------------------------------------------------------
# 1) Evolutionärer Algorithmus (DEAP) - Setup
# -----------------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual(num_points=8):
    """
    Erzeugt ein generisches Polygon mit 'num_points' Punkten.
    Jeder Punkt liegt irgendwo in der Kanalmitte (x ~ [0.8..2.2], y ~ [-0.4..0.4]),
    sodass genug Varianz vorhanden ist.
    """
    polygon = []
    for _ in range(num_points):
        x_coord = np.random.uniform(0.8, 2.2)
        y_coord = np.random.uniform(-0.4, 0.4)
        polygon.append((x_coord, y_coord))
    return polygon

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------------------------------------------------------
# 2) Navier-Stokes-Simulation (vereinfacht) + CW-Berechnung
# -----------------------------------------------------------------------------
def simulate_navier_stokes(shape_points,
                           nx=60, ny=40,
                           Re=50.0,
                           U_in=1.0,
                           max_iter=3000,
                           tol=1e-5):
    """
    Führt eine einfache 2D-Navier-Stokes-Simulation (Projection-Methode)
    durch und iteriert in der Zeit, bis ein (annähernd) stationärer Zustand
    erreicht ist oder max_iter überschritten ist.
    """
    x_min, x_max = 0.0, 3.0
    y_min, y_max = -0.5, 0.5
    Lx = x_max - x_min
    Ly = y_max - y_min

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    nu = U_in / Re

    xg = np.linspace(x_min, x_max, nx)
    yg = np.linspace(y_min, y_max, ny)

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Einströmprofil
    u[:, 0] = U_in

    # Hindernis: bounding box aus shape_points
    shp_x = [pt[0] for pt in shape_points]
    shp_y = [pt[1] for pt in shape_points]
    obs_xmin, obs_xmax = min(shp_x), max(shp_x)
    obs_ymin, obs_ymax = min(shp_y), max(shp_y)

    XX, YY = np.meshgrid(xg, yg)  # (ny, nx)
    obstacle_mask = (
        (XX >= obs_xmin) & (XX <= obs_xmax) &
        (YY >= obs_ymin) & (YY <= obs_ymax)
    )

    def apply_boundary_conditions(u_, v_, p_):
        # Einlass
        u_[:, 0] = U_in
        v_[:, 0] = 0.0
        # Auslass
        u_[:, -1] = u_[:, -2]
        v_[:, -1] = v_[:, -2]
        p_[:, -1] = p_[:, -2]
        # Oben/unten
        u_[0, :] = 0.0
        v_[0, :] = 0.0
        u_[-1, :] = 0.0
        v_[-1, :] = 0.0
        # Hindernis
        u_[obstacle_mask] = 0.0
        v_[obstacle_mask] = 0.0

    dt = 0.05 * min(dx, dy) / U_in

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        u_old = u.copy()
        v_old = v.copy()

        ue = u.copy()
        ve = v.copy()

        lap_u = np.zeros_like(u)
        lap_v = np.zeros_like(u)
        dudx  = np.zeros_like(u)
        dudy  = np.zeros_like(u)
        dvdx  = np.zeros_like(u)
        dvdy  = np.zeros_like(u)

        lap_u[1:-1, 1:-1] = (
              (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
            + (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        )
        lap_v[1:-1, 1:-1] = (
              (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
            + (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        )

        dudx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)
        dudy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy)
        dvdx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)
        dvdy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)

        ue[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            - dt * (u[1:-1, 1:-1]*dudx[1:-1, 1:-1] +
                    v[1:-1, 1:-1]*dudy[1:-1, 1:-1])
            + nu * dt * lap_u[1:-1, 1:-1]
        )
        ve[1:-1, 1:-1] = (
            v[1:-1, 1:-1]
            - dt * (u[1:-1, 1:-1]*dvdx[1:-1, 1:-1] +
                    v[1:-1, 1:-1]*dvdy[1:-1, 1:-1])
            + nu * dt * lap_v[1:-1, 1:-1]
        )

        if (not np.isfinite(ue).all()) or (not np.isfinite(ve).all()):
            return 1e6, u, v, p

        p_old = p.copy()
        for _ in range(50):
            p[1:-1, 1:-1] = (
                (p_old[1:-1, 2:] + p_old[1:-1, :-2] +
                 p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / 4.0
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

        u[1:-1, 1:-1] = ue[1:-1, 1:-1] - dt*(p[1:-1, 2:] - p[1:-1, :-2])/(2*dx)
        v[1:-1, 1:-1] = ve[1:-1, 1:-1] - dt*(p[2:, 1:-1] - p[:-2, 1:-1])/(2*dy)

        apply_boundary_conditions(u, v, p)

        if (not np.isfinite(u).all()) or (not np.isfinite(v).all()):
            return 1e6, u, v, p

        diff_u = np.linalg.norm(u - u_old, ord=2)
        diff_v = np.linalg.norm(v - v_old, ord=2)
        if diff_u < tol and diff_v < tol:
            break

    fy = (y_max - y_min)
    fx = np.sum(p[:, 0] - p[:, -1]) * (fy / ny)

    cross_area = (obs_ymax - obs_ymin)
    if cross_area < 1e-10:
        cross_area = 1e-10

    cw = fx / (0.5 * (U_in**2) * cross_area)

    return abs(cw), u, v, p

def calculate_cw_ns(shape_points, plot_dir=None):
    """
    Führt die Navier-Stokes-Simulation durch und gibt den CW-Wert zurück.
    """
    cw, u, v, p = simulate_navier_stokes(shape_points,
                                         nx=60, ny=40,
                                         Re=50.0,   
                                         U_in=1.0,
                                         max_iter=3000,
                                         tol=1e-5)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        x_min, x_max = 0.0, 3.0
        y_min, y_max = -0.5, 0.5
        nx_ = u.shape[1]
        ny_ = u.shape[0]
        xg = np.linspace(x_min, x_max, nx_)
        yg = np.linspace(y_min, y_max, ny_)
        X, Y = np.meshgrid(xg, yg)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, p, levels=20)
        plt.colorbar(label='Druck')
        plt.title(f"Druckfeld (CW={cw:.2f})")
        plt.xlabel("x")
        plt.ylabel("y")

        speed = np.sqrt(u**2 + v**2)
        plt.subplot(1, 2, 2)
        plt.streamplot(X, Y, u, v, density=1.2, color='k')
        plt.contourf(X, Y, speed, alpha=0.5)
        plt.colorbar(label='|u|')
        plt.title("Geschwindigkeitsfeld")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "ns_solution.png"))
        plt.close()

    return cw

# -----------------------------------------------------------------------------
# 3) Genetische Operatoren
# -----------------------------------------------------------------------------
def mutate_polygon(individual, indpb=0.3):
    """
    Aggressive Mutation:
    - Höhere Probability (indpb=0.3)
    - Größere Verschiebung: ±0.5 in x, ±0.25 in y
    """
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            x_shift = np.random.uniform(-0.5, 0.5)
            y_shift = np.random.uniform(-0.25, 0.25)
            individual[i] = (individual[i][0] + x_shift,
                             individual[i][1] + y_shift)
    return (individual,)

def crossover_polygon(ind1, ind2, indpb=0.7):
    """
    Cross-Over mit erhöhter Wahrscheinlichkeit.
    """
    for i in range(len(ind1)):
        if np.random.rand() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

toolbox.register("evaluate", lambda ind: (calculate_cw_ns(ind),))
toolbox.register("mate", crossover_polygon)
toolbox.register("mutate", mutate_polygon, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------------------------------------------------------
# 4) Hilfsfunktionen für Visualisierung & Speicherung
# -----------------------------------------------------------------------------
def save_generation(population, gen):
    """
    Speichert pro Individuum (Struktur) einer Generation:
     - form.png (Polygon-Plot)
     - ns_solution.png (Druck/Geschwindigkeit)
    """
    if isinstance(gen, int):
        gen_dir = f"results/gen_{gen:02d}"
    else:
        gen_dir = f"results/{gen}"
    os.makedirs(gen_dir, exist_ok=True)
    
    for idx, ind in enumerate(population):
        ind_dir = os.path.join(gen_dir, f"ind_{idx:02d}")
        os.makedirs(ind_dir, exist_ok=True)

        # Polygon-Plot
        plt.figure(figsize=(6, 4))
        poly_x, poly_y = zip(*(ind + [ind[0]]))
        plt.plot(poly_x, poly_y, '-o')
        plt.fill(poly_x, poly_y, alpha=0.3)
        plt.xlim(0, 3)
        plt.ylim(-0.6, 0.6)
        plt.title(f"Struktur (CW: {ind.fitness.values[0]:.2f})")
        plt.savefig(os.path.join(ind_dir, "form.png"))
        plt.close()

        # Strömungsfeld neu berechnen + abspeichern
        calculate_cw_ns(ind, plot_dir=ind_dir)

def plot_progress(logbook):
    """Zeichnet den Verlauf (Avg & Min) über die Generationen als PNG."""
    gens = logbook.select("gen")
    avg_vals = logbook.select("avg")
    best_vals = logbook.select("min")

    plt.figure(figsize=(8, 5))
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
    # GA-Parameter
    pop_size = 20      # <--- Hier wurde die Populationsgröße auf 20 erhöht
    n_generations = 15

    # Population initialisieren
    population = toolbox.population(n=pop_size)

    # Statistik & Log
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    logbook = tools.Logbook()

    for gen in range(n_generations):
        # Fitness berechnen
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Speichern (Form + Flowplots)
        save_generation(population, gen)

        # Logging
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
