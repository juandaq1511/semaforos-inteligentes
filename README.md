# README
- Esta carpeta contiene archivos con distinto propósito. Para generar rutas viables con las tasas deseadas, se tiene que ir al archivo generarutas.py y poner las tasas deseadas y el número de escenarios deseados.
- Al tener los escenarios, se crea una carpeta llamada rutas a la cual están apuntando los demás .py. En la carpeta, se encuentran los escenarios utilizados para la tesis (rutas_bajo es el escenario real y rutas_alto es el escenario real + un 20% de carros)
- Posteriormente, el usuario puede elegir entre poner a entrenar un algoritmo de aprendizaje por refuerzo (PPO, DQN o A2C), correr el Optuna con el fin de encontrar los mejores hiperparámetros o correr un método tradicional con el fin de obtener estadísticas.

# Para entrenar
Se abre el archivo de entorno_sumo.py y se modifica la línea 50 con el fin de generar el nombre del excel con base en el algoritmo utilizado. Por último, se accede al archivo correspondiente a ese algoritmo y se modifican los parámetros e hiperparámetros según el usuario y se pone a correr el código.
# Para correr el Optuna
Se abre el archivo Optuna.py y se definen los parámetros deseados
# Para correr un método tradicional
Se abre el archivo correspondiente al método deseado. Posteriormente, se le indica el nombre deseado al excel generado y se definen el número de episodios que se quiere correr el modelo.
