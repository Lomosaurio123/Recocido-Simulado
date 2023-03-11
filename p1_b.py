# Recocido simulado para la segunda parte

import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo de las matrices de flujo y distancia


# Leer el archivo y asignar los valores a una matriz
data = np.genfromtxt("matricesProblemaQAP/ajuste.dat", dtype=int, delimiter=None, skip_header=1)

# Obtener la cantidad de columnas
num_columns = int(np.genfromtxt("matricesProblemaQAP/ajuste.dat", dtype=int, delimiter=None, max_rows=1))

# Separar las dos matrices en variables distintas
flujo = data[:num_columns, :]
distancias = data[num_columns:, :]

# Realizar la función de evaluación

def evaluation(x):

    costo = 0
    
    for i in range( len( flujo ) ):
        for j in range( len( distancias ) ):

            costo += distancias[i][j] * flujo[ x[i] ][ x[j] ]

    return costo #Parte a optimizar

# Recocido simulado

def rand_solution(num_vars):
    """
    Función para generar una solución aleatoria al problema de asignación cuadrática.
    """
    return np.random.permutation(num_vars)

def neighbor_solution_INTERCAMBIO(x):
    """
    Función para generar una solución vecina de la solución actual. Se intercambian dos elementos aleatorios
    de la solución.
    """
    y = np.copy(x)
    # Obtener dos índices aleatorios distintos
    i, j = np.random.choice(len(x), size=2, replace=False)
    y[i], y[j] = y[j], y[i]
    return y

def neighbor_solution_REVERSION(x):
    """
    Función para generar una solución vecina de la solución actual. Se revierte un subconjunto de la permutación.
    """ 

    y = np.copy(x)
    #Obtenemos los indices de la reversion 
    i, j = np.random.choice(len(x), size=2, replace=False)
    y[i:j] = y[i:j][::-1]

    return y

def neighbor_solution_INSERCION(x):
    """
    Función para generar una solución vecina de la solución actual. Se inserta elemento de la permutación y se desplazan los demás.
    """
    y = np.copy(x)
    # Obtener dos índices aleatorios distintos
    i, j = np.random.choice(len(x), size=2, replace=False)
    # Insertar el elemento x[j] en la posición i y desplazar los demás elementos
    if i < j:
        y[i+1:j+1] = x[i:j]
    else:
        y[j:i] = x[j+1:i+1]
    y[i] = x[j]
    return y


def acceptance_probability(x, y, function, temp):
    """
    Función para calcular la probabilidad de aceptar una solución vecina. Si la solución vecina es mejor que la
    solución actual, se acepta. En caso contrario, se acepta con una probabilidad decreciente en función de la
    temperatura y la diferencia entre la función evaluada en la solución actual y la solución vecina.
    """
    fx = function(x)
    fy = function(y)
    if fy < fx:
        return True
    p = np.exp(-(fy - fx)/temp)
    return np.random.random() < p

def decrease_temp(beta, temp):
    """
    Función para reducir la temperatura en cada iteración.
    """
    return temp *  beta

def plot_convergence(states):
    """
    Función para graficar la convergencia del algoritmo.
    """
    plt.plot(states)
    plt.title("Convergencia de recocido simulado")
    plt.xlabel("Iteración")
    plt.ylabel("Valor de la función")
    plt.show()

def simulated_annealing(temp, max_steps, num_vars, function):
    x = rand_solution(num_vars)
    states = []
    beta = 0.95
    for step in range(max_steps):
        #Seleccionamos la semilla
        np.random.RandomState(12345)
        #Generamos los vecinos
        y = neighbor_solution_INTERCAMBIO(x)
        if acceptance_probability(x, y, function, temp):
            x = y
        state = function(x)
        states.append(state)
        print("X: ", x, "FUNCTION VALUE ==> ", state, " Temp: ", temp)
        temp = decrease_temp(beta, temp)
    return x, function(x), states

# Ejemplo de uso
num_vars = num_columns
max_steps = 10000
temp = 100
solution = simulated_annealing(temp, max_steps, num_vars, evaluation)
print("Solución: ", solution[0])
print("Valor de la función: ", solution[1])
plot_convergence(solution[2])