import numpy as np
import os
import random
import math

def generar_rutas(seed, output_path):
    np.random.seed(seed)
    random.seed(seed)

    duracion_simulacion = 1800  
    max_vehiculos = 2500      
    aumento = 2

    rutas_explicitas = {
        ("sur", "oriente"): ["-1043296296#1", "-1061719283#1", "-1013171321#1", "-958496038#1", "-958496040#1"],
        ("sur", "occidente"): ["-1043296296#1", "-1061719283#1", "-1013171321#1", "-958496038#1", "958496039#1"],
        ("norte", "sur"): ["1013846480#1", "89987002#0", "89987002#2", "-958496037#0"],
        ("norte", "oriente"): ["1013846480#1", "89987002#0", "89987002#2", "-958496040#1"],
        ("norte", "occidente"): ["1013846480#1", "89987002#0", "89987002#2", "958496039#1"],
        ("oriente", "sur"): ["25444693#0", "25444693#1", "25444693#3", "958496040#1", "-958496037#0"],
        ("oriente", "occidente"): ["25444693#0", "25444693#1", "25444693#3", "958496040#1", "958496039#1"],
        ("occidente", "sur"): ["-89986459#3", "-89986459#2", "-89986459#0", "-958496039#1", "-958496037#0"],
        ("occidente", "oriente"): ["-89986459#3", "-89986459#2", "-89986459#0", "-958496039#1", "-958496040#1"]
    }

    tasas = {
        "sur": 0.1518248,
        "oriente": 0.0803571,
        "occidente": 0.0574713
    }

    lognormal_params = {
        "norte": (3.06 - math.log(aumento), 0.8899785)  # Ajuste para mayor tasa
    }

    vehiculos = []

    for grupo_origen in ["sur", "norte", "oriente", "occidente"]:
        if grupo_origen in tasas:
            escala = 1 / (tasas[grupo_origen] * aumento)
            tiempos = np.cumsum(np.random.exponential(scale=escala, size=max_vehiculos))
        else:
            media, sigma = lognormal_params[grupo_origen]
            tiempos = np.cumsum(np.random.lognormal(mean=media, sigma=sigma, size=max_vehiculos))

        tiempos = tiempos[tiempos <= duracion_simulacion]

        for i, t in enumerate(tiempos):
            posibles_destinos = [g2 for (g1, g2) in rutas_explicitas if g1 == grupo_origen]
            grupo_destino = random.choice(posibles_destinos)
            ruta = rutas_explicitas[(grupo_origen, grupo_destino)]
            vehiculos.append((t, grupo_origen, i, ruta))

    vehiculos.sort(key=lambda x: x[0])

    with open(output_path, "w") as f:
        f.write("<routes>\n")
        f.write('<vType id="car" accel="2" decel="4.5" length="5" maxSpeed="25"/>\n')
        for t, grupo, i, ruta in vehiculos:
            f.write(f'<vehicle id="{grupo}{i}" type="car" depart="{t:.2f}" arrivalPos="max">\n')
            f.write(f'    <route edges="{" ".join(ruta)}"/>\n')
            f.write('</vehicle>\n')
        f.write("</routes>\n")

if __name__ == "__main__":
    rutas_dir = "Simulación/rutas" 
    os.makedirs(rutas_dir, exist_ok=True)

    for i in range(1, 10001):
        seed = random.randint(0, 999999)
        nombre_archivo = os.path.join(rutas_dir, f"rutas_{i:03d}.rou.xml")
        generar_rutas(seed, nombre_archivo)
        print(f"Generado: {nombre_archivo} (semilla: {seed})")

