import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pygame
import os

def run(is_training=True, render=False):
    print("Inicializando o ambiente...")
    env = gym.make('CartPole-v1')

    if render:
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('McQueen - Piston Cup Edition')
        clock = pygame.time.Clock()
        background_image = pygame.image.load('fundo.png')

    car_image = pygame.image.load('McQueen.png')
    pole_image = pygame.image.load('pole.png')
    pole_image = pygame.transform.scale(pole_image, (70, 80))

    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        print("Inicializando tabela Q...")
        q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n))
    else:
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1
    discount_factor_g = 0.99

    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()

    rewards_per_episode = []
    i = 0

    print("Iniciando o loop de episódios...")
    while True:
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

            if render:
                render_custom(screen, background_image, car_image, pole_image, state)
                clock.tick(60)

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            print("Condição de parada atingida.")
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    env.close()
    print("Fechando o ambiente...")

    if is_training:
        print("Salvando tabela Q...")
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Cria o diretório se não existir
    if not os.path.exists('grafico'):
        os.makedirs('grafico')

    # Cria o gráfico
    print("Gerando o gráfico...")
    mean_rewards = [np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(i)]
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label='Média de Recompensas por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Média')
    plt.title('Gráfico de Episódio vs Recompensa')
    plt.legend()
    plt.grid(True)
    plt.savefig('11111111111111111.png')

    # Tenta salvar o gráfico e captura erros
    try:
        plt.savefig('grafico/cartpole.png')
        print('Gráfico salvo como grafico/cartpole.png')
    except Exception as e:
        print(f'Erro ao salvar o gráfico: {e}')

    if render:
        pygame.quit()

def render_custom(screen, background_image, car_image, pole_image, state):
    screen.blit(background_image, (0, 0))

    screen_width, screen_height = screen.get_size()
    car_width, car_height = car_image.get_size()
    pole_width, pole_height = pole_image.get_size()

    cart_position = screen_width / 2 + state[0] * (screen_width / 2 / 2.4)
    car_x = cart_position - car_width / 2
    car_y = screen_height * 0.8

    pole_x = cart_position - pole_width / 2
    pole_y = screen_height * 0.71
    pole_angle = np.degrees(state[2])

    screen.blit(car_image, (car_x, car_y))

    rotated_pole = pygame.transform.rotate(pole_image, pole_angle)
    new_pole_rect = rotated_pole.get_rect(center=(cart_position, pole_y))
    screen.blit(rotated_pole, new_pole_rect.topleft)

    pygame.display.flip()

if __name__ == '__main__':
    run(is_training=False, render=True)
