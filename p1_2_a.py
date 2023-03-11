import math
import random 
import numpy as np
import matplotlib.pyplot as plt

def michalewicz(x, m=10):
    """Función de Michalewicz"""
    return -sum([math.sin(x[i]) * math.sin((i+1)*x[i]**2/math.pi)**(2*m) for i in range(len(x))])

# Función para graficar temperaturas y convergencia

def plot_convergenceAndTemperature(states, temps):
    """
    Función para graficar la convergencia del algoritmo.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    ax1.plot(states)
    ax1.set_title("Convergencia de recocido simulado")
    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Valor de la función")
    ax1.grid()
    
    ax2.plot(temps)
    ax2.set_title("Temperatura de recocido simulado")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Temperatura")
    ax2.grid()
    
    plt.tight_layout()
    plt.show()

# Funciones de decrecimiento de temperatura

def decrease_temp_LINEAL(beta, temp):
    return temp - beta

def decrease_temp_GEOMETRIC(beta, temp):
    return temp * beta

def decrease_temp_LOGARITHMIC(K, temp):

    return ( temp / ( math.log( K + 1 ) ) ) 

def decrease_temp_HYBRID(beta, maxSteps, K, temp):

    if K == 0 : K = 0.00000000001

    if K <= int(maxSteps * 0.3): 
        return ( K / ( K + 1 ) ) * temp
    else: 
        return decrease_temp_GEOMETRIC( beta, temp )
    
def decrease_temp_EXPONENTIAL(beta, temp):
    return temp / ( 1 + ( beta * temp ) )

def simulated_annealing(initial_solution, initial_temperature, cooling_factor, stopping_temperature, num_vars):
    """Algoritmo del Recocido Simulado"""
    
    # Inicializamos la solución actual y la mejor solución
    current_solution = initial_solution
    best_solution = initial_solution
    
    # Inicializamos la energía actual y la mejor energía
    current_energy = michalewicz(current_solution)
    best_energy = current_energy
    
    # Inicializamos la temperatura
    temperature = initial_temperature
    
    # Inicializamos el contador de iteraciones
    iteration = 0

    # Establecemos la semilla

    np.random.seed( 54321 )

    # Inicializamos los estados

    states = []
    temperatures = []
    
    # Iteramos hasta alcanzar la temperatura de parada
    while temperature > stopping_temperature:
        
        # Generamos una nueva solución vecina
        neighbor_solution = [current_solution[i] + random.uniform(-1, 1) for i in range(num_vars)]
        
        # Limitamos los valores de la solución al rango entre 0 y pi
        neighbor_solution = np.clip(neighbor_solution, 0, math.pi)
        
        # Evaluamos la energía de la nueva solución vecina
        neighbor_energy = michalewicz(neighbor_solution)
        
        # Calculamos la diferencia de energía
        energy_delta = neighbor_energy - current_energy
        
        # Si la nueva solución vecina es mejor que la actual, la aceptamos
        if energy_delta < 0:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
            
            # Si la nueva solución vecina es la mejor encontrada, la actualizamos
            if neighbor_energy < best_energy:
                best_solution = neighbor_solution
                best_energy = neighbor_energy
                
        # Si la nueva solución vecina es peor que la actual, la aceptamos con una probabilidad
        # que depende de la temperatura y la diferencia de energía
        else:
            probability = math.exp(-energy_delta/temperature)
            if random.random() < probability:
                current_solution = neighbor_solution
                current_energy = neighbor_energy
        
        states.append( current_energy )
                
        # Actualizamos la temperatura
        iteration += 1
        temperature = decrease_temp_GEOMETRIC( cooling_factor, temperature )
        temperatures.append( temperature )

        # # Variante para retornar la temperatura a su estado original

        # if( iteration == 1000 * num_vars ) : break
        
        # if( iteration % 100 == 0 ) : temperature = 100
        
    return best_solution, best_energy, states, temperatures

# Definimos los parámetros
num_vars = 2
initial_solution = [random.uniform(0, math.pi)] * num_vars
temperature = 100
cooling_rate = 0.999
stopping_temperature = 1e-8

# Ejecutamos el algoritmo del Recocido Simulado
solution = simulated_annealing(initial_solution, temperature, cooling_rate, stopping_temperature, num_vars)

# Imprimimos los resultados
print("Mejor solución encontrada:", solution[0])
print("Mejor energía encontrada:", solution[1])
plot_convergenceAndTemperature(solution[2], solution[3])