import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pygame

def run(is_training=True, render=False):

    env = gym.make('CartPole-v1')

    # Inicializa o Pygame para renderização
    if render:
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('McQueen - Piston Cup Edition')
        clock = pygame.time.Clock()

        # Carrega a imagem de fundo
        background_image = pygame.image.load('fundo.png')

    # Carrega as imagens do carro e do poste
    car_image = pygame.image.load('McQueen.png')
    pole_image = pygame.image.load('pole.png')
    pole_image = pygame.transform.scale(pole_image, (70, 80))  # Ajusta o tamanho do poste

    # Divide a posição, velocidade, ângulo do poste e velocidade angular em segmentos
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n))  # Inicializa um array 11x11x11x11x2
    else:
        with open('sarsa.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1  # alpha ou taxa de aprendizado
    discount_factor_g = 0.99  # gamma ou fator de desconto

    epsilon = 1  # 1 = 100% de ações aleatórias
    epsilon_decay_rate = 0.00001  # taxa de decaimento do epsilon
    rng = np.random.default_rng()  # gerador de números aleatórios

    rewards_per_episode = []

    i = 0

    while True:
        state = env.reset()[0]  # Posição inicial, velocidade inicial sempre 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False  # Verdadeiro quando atinge o objetivo

        rewards = 0

        # Seleciona a primeira ação usando a política epsilon-greedy
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state_p, state_v, state_a, state_av, :])

        while not terminated and rewards < 10000:
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training and rng.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q[new_state_p, new_state_v, new_state_a, new_state_av, :])

            if is_training:
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * q[new_state_p, new_state_v, new_state_a, new_state_av, next_action] - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
            action = next_action

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
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()

    # Salva a tabela Q em um arquivo
    if is_training:
        with open('sarsa.pkl', 'wb') as f:
            pickle.dump(q, f)

    mean_rewards = [np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(i)]
    plt.plot(mean_rewards)
    plt.savefig('cartpole.png')

    if render:
        pygame.quit()

def render_custom(screen, background_image, car_image, pole_image, state):
    # Função de renderização personalizada para desenhar as imagens do carro e do poste no fundo
    screen.blit(background_image, (0, 0))  # Desenha a imagem de fundo

    screen_width, screen_height = screen.get_size()
    car_width, car_height = car_image.get_size()
    pole_width, pole_height = pole_image.get_size()

    # Calcula a posição do carro na tela
    cart_position = screen_width / 2 + state[0] * (screen_width / 2 / 2.4)
    car_x = cart_position - car_width / 2
    car_y = screen_height * 0.8  # 80% do topo

    # Calcula a posição e o ângulo do poste
    pole_x = cart_position - pole_width / 2
    pole_y = screen_height * 0.71
    pole_angle = np.degrees(state[2])  # Converte o ângulo do poste de radianos para graus

    # Desenha a imagem do carro
    screen.blit(car_image, (car_x, car_y))

    # Desenha a imagem do poste com rotação
    rotated_pole = pygame.transform.rotate(pole_image, pole_angle)
    new_pole_rect = rotated_pole.get_rect(center=(cart_position, pole_y))
    screen.blit(rotated_pole, new_pole_rect.topleft)

    pygame.display.flip()

if __name__ == '__main__':
    run(is_training=False, render=True)
