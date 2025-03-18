import random
import matplotlib.pyplot as plt

def monotone_chain(points):
    """
    Erzeugt die konvexe Hülle einer 2D-Punktwolke (Monotone Chain Algorithm).
    Punkte und Rückgabe: Liste von (x,y)-Tupeln in Gegen-Uhrzeiger-Richtung.
    """
    # Sortiere die Punkte nach (x, dann y)
    pts = sorted(points)
    
    # Hilfsfunktion: Kreuzprodukt 2D, det([OA, OB])
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    # Untere Kante
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Obere Kante
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Entferne den jeweils letzten Punkt (Dopplung)
    lower.pop()
    upper.pop()

    # Konkatenate untere + obere Kante => geschlossene Hülle (CCW)
    return lower + upper

def main():
    # 1) Erzeuge zufällige Punkte
    random.seed(42)
    points = [(random.uniform(0,10), random.uniform(0,10)) for _ in range(30)]
    
    # 2) Berechne die konvexe Hülle
    hull = monotone_chain(points)
    
    # 3) Visualisierung
    # (ein Diagramm ohne spezielle Farben/Styles)
    plt.figure()  # eigener Plot
    
    # Punktwolke
    xs_pts, ys_pts = zip(*points)
    plt.scatter(xs_pts, ys_pts)         # streue alle Punkte
    
    # Konvexe Hülle
    # (hull plus erster Punkt erneut, um das Polygon zu schließen)
    hull_closed = hull + [hull[0]]
    xs_hull, ys_hull = zip(*hull_closed)
    plt.plot(xs_hull, ys_hull)         # plotte die Hülle

    plt.title("Zufällige Punktwolke und ihre Konvexe Hülle")
    plt.show()

if __name__ == "__main__":
    main()
