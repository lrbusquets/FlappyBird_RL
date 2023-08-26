import pygame
from pygame.locals import *
import copy

pygame.init()
clock = pygame.time.Clock()
fps = 60

factor = 0.333
screen_width = 1080 * factor
screen_height = 1920 * factor

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy bird')

# define font & colour for text
font = pygame.font.SysFont('Bauhaus 93', 60)
white = (255, 255, 255)


# load images
bg = pygame.image.load('images/bg.png')
ground_img = pygame.image.load('images/ground.png')
button_img = pygame.image.load('images/restart.png')



#define game variables

gravity = 0.55
v_terminal = 8  # desired terminal velocity
k = gravity/(v_terminal**2)  # required rho*CD*A/2 so that the terminal velocity is the wanted one

scroll_speed = 4
floor_location = 0.9*screen_height

pipe_gap = 120
pipe_freq = 1500  # [ms]

game_params = {
    'gravity': gravity,
    'k': k,
    'scroll_speed': scroll_speed,
    'floor_location': floor_location,
    'pipe_gap': pipe_gap,
    'pipe_freq': pipe_freq,
}