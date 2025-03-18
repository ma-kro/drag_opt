

Wichtige Aspekte
convex_hull(...):

Wir verwenden den Monotone-Chain-Algorithmus, der in 
𝑂
(
𝑛
log
⁡
𝑛
)
O(nlogn) die Hülle berechnet. Für 10–20 Punkte pro Individuum ist das sehr schnell.
Danach hast Du die Eckpunkte in Gegen-Uhrzeiger-Richtung (CCW), was für Plotten etc. angenehm ist.
create_individual und mutate_base_points:

Wir arbeiten mit einer Liste von Basis-Punkten. Dank convex_hull(...) ist das resultierende Polygon stets konvex.
Bounding Box in der Strömung

Auch jetzt bist Du weiterhin auf bounding box. Das heißt, alle konvexen Feinheiten werden leider ignoriert und die Strömung „sieht“ nur ein Rechteck.
Willst Du wirklich unterschiedliche konvexe Formen in der Strömung simulieren, müsstest Du pro Gitterpunkt fragen, ob er innerhalb des Polygons liegt (nicht nur „innerhalb der bounding box“).
Mögliche Verbesserungen

Größere Population, mehr Generationen, Upwind-Schema bei hohem Re, etc. – genau wie vorher.
Für Dein Ziel, dass die Struktur wirklich „stromlinienförmiger“ wird, solltest Du – wie erklärt – ein echtes Punkt-in-Polygon-Verfahren implementieren, sonst bekommst Du in der Simulation immer nur eckige Hindernisse.


Zusammenfassung
Nur konvexe Strukturen: Wir speichern die Basis-Punkte, berechnen daraus bei Bedarf eine konvexe Hülle.
Gleichbleibende bounding box-Simulation: Noch keine wirklich unterschiedlichen Hindernisformen, da wir noch kein echtes Inside/Outside-Verfahren haben.




Wichtiger Hinweis:
Da wir in der Navier-Stokes-Simulation immer noch nur die Bounding Box des Polygons als Hindernis verwenden, kann es sein, dass alle konvexen Feinheiten im Inneren ignoriert werden. Du bekommst zwar konvexe Polygone, aber solange Du kein echtes Punkt-in-Polygon-Verfahren in der Strömungssimulation implementierst, wirst Du weiterhin nur Quader-Masken sehen. Der Code zeigt aber exemplarisch, wie man reine konvexe Polygone erzwingen kann.