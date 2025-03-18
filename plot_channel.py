import matplotlib.pyplot as plt
import numpy as np

def draw_channel_with_structure():
    # 1) Erstelle eine Figur
    plt.figure(figsize=(8, 3))
    
    # 2) Strömungskanal: x=[0..3], y=[-0.5..0.5]
    # Zeichne die Kanalgrenzen:
    plt.axhline(y=-0.5, color='black', linestyle='--')  # untere Grenze
    plt.axhline(y= 0.5, color='black', linestyle='--')  # obere Grenze
    plt.axvline(x= 0.0, color='gray', linestyle=':')    # linker Bereich
    plt.axvline(x= 3.0, color='gray', linestyle=':')    # rechter Bereich
    
    # 3) Beispiel-Struktur:
    #   ein Rechteck (von x=1.0 bis x=2.0, y=-0.3 bis y=0.3) in der Kanalmitte
    rect_x = [1.0, 2.0, 2.0, 1.0, 1.0]
    rect_y = [0.3, 0.3,-0.3,-0.3, 0.3]
    plt.plot(rect_x, rect_y, '-o', label='Struktur')
    
    # 4) Pfeile für Ein- und Ausströmung
    #    Einströmung (links, x=0)
    plt.annotate('Einlass\n(Flow In)', xy=(0,-0.0), xytext=(-0.3, 0.0),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 ha='right', va='center')
    #    Ausströmung (rechts, x=3)
    plt.annotate('Auslass\n(Flow Out)', xy=(3,0.0), xytext=(3.3, 0.0),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='left', va='center')
    
    # 5) Achsenbeschriftungen & Grenzen
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.8, 0.8)
    plt.xlabel('x-Richtung')
    plt.ylabel('y-Richtung')
    plt.title('Schematischer Strömungskanal (2D) mit Hindernis')
    plt.legend()
    
    # 6) Anzeigen oder Speichern
    plt.tight_layout()
    plt.show()
    # plt.savefig("kana mit struktur.png")

# Aufruf:
if __name__ == "__main__":
    draw_channel_with_structure()
