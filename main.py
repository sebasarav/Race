import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.track_surface = self.create_track()
        self.create_rects()
        self.checkpoints = self.create_checkpoints()
        # Añadir la línea de meta como un rectángulo para detectar colisiones
        outer_track_width = self.width - 200
        outer_track_height = self.height - 200
        finish_line_x = self.width // 2 - outer_track_width // 2 + 530
        finish_line_y = self.height // 2 + outer_track_height // 2 - 200
        self.finish_line = pygame.Rect(
            finish_line_x - 10,  # x initial
            finish_line_y - 5,    # y initial (a bit higher for better detection)
            20,                  # width
            10                   # height
        )
    def check_finish_line(self, vehicle):
        # Create a rectangle representing the finish line
        finish_line_rect = pygame.Rect(
            self.finish_line.x,
            self.finish_line.y,
            vehicle.finish_line_width,
            vehicle.finish_line_height
        )

        # Check if the vehicle is overlapping the finish line rectangle
        if finish_line_rect.colliderect(vehicle.position[0] - vehicle.finish_line_width//2,
                                        vehicle.position[1] - vehicle.finish_line_height//2,
                                        vehicle.finish_line_width,
                                        vehicle.finish_line_height):
            if not vehicle.crossed_finish_line:
                vehicle.crossed_finish_line = True
                vehicle.laps_completed += 1
                return True
        else:
            vehicle.crossed_finish_line = False

        return False
        
    def create_checkpoints(self):
        # Crear checkpoints en la pista verde (entre los rectángulos exterior e interior)
        checkpoints = []
        num_checkpoints = 20
        
        # Obtener dimensiones medias entre los rectángulos exterior e interior
        outer_track_width = self.width - 200
        outer_track_height = self.height - 200
        inner_track_width = self.width - 340
        inner_track_height = self.height - 340
        
        # Calcular el radio medio entre los rectángulos
        radius_x = (outer_track_width + inner_track_width) / 4
        radius_y = (outer_track_height + inner_track_height) / 4
        
        center_x = self.width // 2
        center_y = self.height // 2
        
        for i in range(num_checkpoints):
            angle = (2 * np.pi * i) / num_checkpoints
            
            # Agregar un pequeño offset aleatorio al radio para variar la posición
            # pero manteniendo los checkpoints dentro de la pista verde
            radius_offset = np.random.uniform(-20, 20)
            current_radius_x = radius_x + radius_offset
            current_radius_y = radius_y + radius_offset
            
            x = center_x + current_radius_x * np.cos(angle)
            y = center_y + current_radius_y * np.sin(angle)
            
            # Verificar que el checkpoint está dentro de la pista verde
            point = pygame.math.Vector2(x, y)
            if (self.outer_track_rect.collidepoint(point) and 
                not self.inner_track_rect.collidepoint(point)):
                checkpoints.append((x, y))
            
        return checkpoints

    def create_track(self):
        track = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        outer_track_width = self.width - 200
        outer_track_height = self.height - 200
        inner_track_width = self.width - 340
        inner_track_height = self.height - 340

        pygame.draw.rect(track, (0, 255, 0), 
                         (self.width // 2 - outer_track_width // 2, 
                          self.height // 2 - outer_track_height // 2, 
                          outer_track_width, outer_track_height), 
                         border_radius=50)

        pygame.draw.rect(track, (0, 0, 0), 
                         (self.width // 2 - inner_track_width // 2, 
                          self.height // 2 - inner_track_height // 2, 
                          inner_track_width, inner_track_height), 
                         border_radius=50)

        self.start_line_x = self.width // 2 - outer_track_width // 2 + 400
        self.start_line_y = self.height // 2 - outer_track_height // 2 + 0.9
        start_line_y_end = self.start_line_y + 60
        pygame.draw.line(track, (128, 0, 128), 
                         (self.start_line_x, self.start_line_y), 
                         (self.start_line_x, start_line_y_end + 10), 10)

        finish_line_x = self.width // 2 - outer_track_width // 2 + 600
        finish_line_y = self.height // 2 + outer_track_height // 2 - 200
        pygame.draw.line(track, (255, 0, 0), 
                         (finish_line_x , finish_line_y), 
                         (finish_line_x - 70, finish_line_y), 10)

        return track

    def create_rects(self):
        inner_track_width = self.width - 340
        inner_track_height = self.height - 340
        outer_track_width = self.width - 200
        outer_track_height = self.height - 200

        self.inner_track_rect = pygame.Rect(
            self.width // 2 - inner_track_width // 2, 
            self.height // 2 - inner_track_height // 2, 
            inner_track_width, 
            inner_track_height
        )
        
        self.outer_track_rect = pygame.Rect(
            self.width // 2 - outer_track_width // 2, 
            self.height // 2 - outer_track_height // 2, 
            outer_track_width, 
            outer_track_height
        )

    def draw(self, screen):
        screen.blit(self.track_surface, (0, 0))
        self.draw_boundaries(screen)
        # Dibujar la línea de meta
        pygame.draw.rect(screen, (255, 0, 0), self.finish_line, 2)

    def draw_boundaries(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), self.outer_track_rect, 2)
        pygame.draw.rect(screen, (0, 0, 255), self.inner_track_rect, 2)

class Vehicle:
    def __init__(self, start_pos):
        self.start_pos = np.array(start_pos, dtype=float)
        self.checkpoint_radius = 30
        self.last_checkpoint = 0
        self.laps_completed = 0
        self.crossed_finish_line = False  # Nuevo: para controlar el cruce de la línea de meta
        self.reset()
        self.finish_line_width = 20  # Width of the finish line rectangle
        self.finish_line_height = 10 # Height of the finish line rectangle

    def reset(self):
        self.position = self.start_pos.copy()
        self.angle = 180
        self.speed = 0
        self.max_speed = 5
        self.min_speed = -2
        self.acceleration = 0.1
        self.turning_speed = 3
        self.radars = [-60, -30, 0, 30, 60]
        self.radar_distances = [1.0] * len(self.radars)
        self.distance_traveled = 0
        self.last_checkpoint = 0
        self.laps_completed = 0
        self.time_alive = 0
        self.crossed_finish_line = False
        self.last_checkpoint_time = 0

    def check_finish_line(self, track):
        # Create a rectangle representing the finish line
        finish_line_rect = pygame.Rect(
            track.finish_line.x,
            track.finish_line.y,
            self.finish_line_width,
            self.finish_line_height
        )

        # Check if the vehicle is overlapping the finish line rectangle
        if finish_line_rect.colliderect(self.position[0] - self.finish_line_width//2,
                                        self.position[1] - self.finish_line_height//2,
                                        self.finish_line_width,
                                        self.finish_line_height):
            if not self.crossed_finish_line and time.time() - self.last_checkpoint_time > 1:
                self.crossed_finish_line = True
                self.laps_completed += 1
                self.last_checkpoint_time = time.time()  # Nuevo: actualizar el tiempo del último checkpoint
                return True
        else:
            self.crossed_finish_line = False

        return False
        
    def accelerate(self):
        self.speed = min(self.speed + self.acceleration, self.max_speed)
        
    def brake(self):
        self.speed = max(self.speed - self.acceleration, self.min_speed)
        
    def update_position(self):
        self.position[0] += self.speed * np.cos(np.radians(self.angle))
        self.position[1] += self.speed * np.sin(np.radians(self.angle))
        self.time_alive += 1
        if self.speed > 0:
            self.distance_traveled += self.speed

    def check_checkpoint(self, checkpoints):
        current_pos = np.array(self.position)
        next_checkpoint_idx = (self.last_checkpoint + 1) % len(checkpoints)
        next_checkpoint = np.array(checkpoints[next_checkpoint_idx])
        
        distance = np.linalg.norm(current_pos - next_checkpoint)
        
        if distance < self.checkpoint_radius:
            self.last_checkpoint = next_checkpoint_idx
            if next_checkpoint_idx == 0:
                self.laps_completed += 1
            return True
        return False
    

    def update_position(self):
        self.position[0] += self.speed * np.cos(np.radians(self.angle))
        self.position[1] += self.speed * np.sin(np.radians(self.angle))
        self.distance_traveled += self.speed  # Aumenta la distancia recorrida

    def get_radar_distance(self, radar_angle, track):
        max_distance = 100  # Rango máximo de los radares
        step = 5  # Incremento para buscar la distancia

        for dist in range(0, max_distance, step):
            point = self.position + np.array([dist * np.cos(np.radians(self.angle + radar_angle)), 
                                              dist * np.sin(np.radians(self.angle + radar_angle))])

            # Verifica si el punto está fuera del borde exterior o dentro del borde interior
            if not track.outer_track_rect.collidepoint(point) or track.inner_track_rect.collidepoint(point):
                return dist / max_distance  # Normaliza la distancia
        return 1.0  # Retorna 1.0 si no detecta borde dentro del rango

    def update_radars(self, track):
        self.radar_distances = [self.get_radar_distance(radar, track) for radar in self.radars]

    def adjust_angle_based_on_radars(self):
        left_radar = self.radar_distances[0]
        front_left_radar = self.radar_distances[1]
        front_radar = self.radar_distances[2]
        front_right_radar = self.radar_distances[3]
        right_radar = self.radar_distances[4]

        # Si detecta un borde a la izquierda o frente izquierdo, gira a la derecha
        if left_radar < 0.3 or front_left_radar < 0.3:
            self.angle += 5
        # Si detecta un borde a la derecha o frente derecho, gira a la izquierda
        elif right_radar < 0.3 or front_right_radar < 0.3:
            self.angle -= 5
        # Si detecta un borde enfrente, gira a la izquierda (sentido antihorario)
        elif front_radar < 0.3:
            self.angle -= 5

    def check_collision(self, track):
        if not track.outer_track_rect.collidepoint(self.position) or track.inner_track_rect.collidepoint(self.position):
            return True
        return False

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.position.astype(int), 10)
        for radar, distance in zip(self.radars, self.radar_distances):
            end_x = self.position[0] + distance * 100 * np.cos(np.radians(self.angle + radar))
            end_y = self.position[1] + distance * 100 * np.sin(np.radians(self.angle + radar))
            pygame.draw.line(screen, (255, 0, 0), self.position.astype(int), (end_x, end_y), 1)

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ReinforcementLearning:
    def __init__(self):
        # Estado: 5 radares + velocidad + ángulo + posición (x,y)
        self.model = QNetwork(9, 4)  # 4 acciones: (nada, acelerar, frenar, girar)
        self.target_model = QNetwork(9, 4)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayBuffer()
        self.batch_size = 64
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        
        self.update_target_every = 10
        self.target_update_counter = 0

    def get_state(self, vehicle):
        # Normalizar valores para el estado
        normalized_pos = vehicle.position / np.array([800, 600])  # Normalizar por dimensiones de la pantalla
        normalized_speed = vehicle.speed / vehicle.max_speed
        normalized_angle = vehicle.angle % 360 / 360
        
        return np.concatenate([
            vehicle.radar_distances,  # 5 valores
            [normalized_speed],       # 1 valor
            [normalized_angle],       # 1 valor
            normalized_pos            # 2 valores
        ])

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Simulador de Vehículo RL")
    clock = pygame.time.Clock()
    
    track = Track(800, 600)
    vehicle = Vehicle(start_pos=(track.start_line_x, track.start_line_y + 35))
    rl_agent = ReinforcementLearning()
    
    episode = 0
    max_steps_per_episode = 2000
    font = pygame.font.Font(None, 36)  # Para mostrar el contador de vueltas
    
    running = True
    while running:
        episode += 1
        step = 0
        total_reward = 0
        vehicle.reset()
        
        while step < max_steps_per_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
                
            # Obtener estado actual
            vehicle.update_radars(track)
            current_state = rl_agent.get_state(vehicle)
            
            # Obtener acción
            action = rl_agent.get_action(current_state)
            
            # Ejecutar acción
            if action == 0:  # No hacer nada
                pass
            elif action == 1:  # Acelerar
                vehicle.accelerate()
            elif action == 2:  # Frenar
                vehicle.brake()
            elif action == 3:  # Girar
                if vehicle.radar_distances[0] < vehicle.radar_distances[4]:
                    vehicle.angle += vehicle.turning_speed
                else:
                    vehicle.angle -= vehicle.turning_speed
            
            # Actualizar posición
            vehicle.update_position()
            
            # Calcular recompensa
            reward = 0
            
            # Recompensa por velocidad y movimiento forward
            reward += vehicle.speed * 0.1
            
            # Recompensa por checkpoints
            if vehicle.check_checkpoint(track.checkpoints):
                reward += 10
            
            # Verificar si cruza la línea de meta
            if vehicle.check_finish_line(track):
                reward += 50
                print(f"¡Vuelta completada! Vueltas totales: {vehicle.laps_completed}")
            
            # Penalización por colisión
            if vehicle.check_collision(track):
                reward -= 20
                done = True
            else:
                done = False
            
            # Obtener siguiente estado
            vehicle.update_radars(track)
            next_state = rl_agent.get_state(vehicle)
            
            # Guardar experiencia
            rl_agent.memory.push(current_state, action, reward, next_state, done)
            
            # Entrenar
            rl_agent.train(rl_agent.batch_size)
            
            total_reward += reward
            step += 1
            
            # Renderizar
            screen.fill((255, 255, 255))
            track.draw(screen)
            vehicle.draw(screen)
            
            # Dibujar checkpoints
            for checkpoint in track.checkpoints:
                pygame.draw.circle(screen, (0, 255, 0), (int(checkpoint[0]), int(checkpoint[1])), 5)
            
            # Mostrar contador de vueltas
            #laps_text = font.render(f"Vueltas: {vehicle.laps_completed}", True, (0, 0, 0))
            #screen.blit(laps_text, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)
            
            if done:
                break
        
        print(f"Episodio {episode} completado - Pasos: {step}, Recompensa total: {total_reward:.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    main()
