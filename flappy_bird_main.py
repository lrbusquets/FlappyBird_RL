from typing import Any
import pygame
from pygame.locals import *
import random
import time
import numpy as np

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

#define game variables
gravity = 0.55
v_terminal = 10 #8  # desired terminal velocity
k = gravity/(v_terminal**2)  # required rho*CD*A/2 so that the terminal velocity is the wanted one
ground_scroll = 0
scroll_speed = 4
floor_location = 0.9*screen_height
flying = False
game_over = False
pipe_gap = 115
pipe_freq = 1500  # [ms]
last_pipe = pygame.time.get_ticks() - pipe_freq
score = 0
pass_pipe = False

# load images
bg = pygame.image.load('images/bg.png')
ground_img = pygame.image.load('images/ground.png')
button_img = pygame.image.load('images/restart.png')

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x,y))

def reset_game():
    pipe_group.empty()  # delete everything
    flappy.rect.x = int(screen_width/6)
    flappy.rect.y = int(screen_height/2)
    score = 0

    return score

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0  # speed of animation
        for num in range(1,4):
            img = pygame.image.load(f'images/bird{num}.png')
            self.images.append(img)

        self.image = self.images[self.index]
        self.rect = self.image.get_rect()  #rectangle
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False

    def update(self):
        
        if flying == True:
            # gravity
            self.vel += gravity - k * self.vel * np.abs(self.vel)
            #if self.vel > 8:  # duh, revise that
            #    self.vel = 8
            if self.rect.bottom < floor_location:
                self.rect.y += int(self.vel)

        if game_over == False:

            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:  # 0-th index for left click
                self.vel = -8.5  # -7.5
                self.clicked = True

            if pygame.mouse.get_pressed()[0] == 0:  # 0-th index for left click
                self.clicked = False

            # handle the animation
            self.counter += 1
            flap_cooldown = 5

            if self.counter > flap_cooldown:
                self.counter = 0
                self.index += 1
                if self.index >= len(self.images):
                    self.index = 0
            self.image = self.images[self.index]

            # rotatae the bird
            self.image = pygame.transform.rotate(self.images[self.index], self.vel * -3)  #ccw by default

        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)  #ccw by default

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/pipe.png')
        self.rect = self.image.get_rect()
        # position 1 is from the top, -1 is from the bottom
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap/2)]
        if position == -1:
            self.rect.topleft = [x, y + int(pipe_gap/2)]

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            self.kill()

class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):

        action = False
        
        # get mouse position
        pos = pygame.mouse.get_pos()

        # check if mouse is over button
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        # draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))
        return action


bird_group = pygame.sprite.Group()
pipe_group = pygame.sprite.Group()

flappy = Bird(int(screen_width/6), int(screen_height/2))  # initial position of bird

bird_group.add(flappy)

button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)

run = True
while run:

    clock.tick(fps)

    # draw background
    screen.blit(bg, (0,-100))   # image drawn from top-left corner

    bird_group.draw(screen)
    bird_group.update()

    pipe_group.draw(screen)

    screen.blit(ground_img, (ground_scroll, floor_location))

    # check score
    if len(pipe_group) > 0:
        if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left\
            and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right\
            and pass_pipe == False:
            pass_pipe = True

        if pass_pipe == True:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                score += 1
                pass_pipe = False

    draw_text(str(score), font, white, int(screen_width/2), int(0.1*screen_height))

    # check for collision
    if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0: # "due kill args": kill bird and pipe
        game_over = True

    # check if bird has hit ground
    if flappy.rect.bottom > floor_location:
        game_over = True
        flying = False

    if game_over == False and flying == True:

        # generate new pipes
        time_now = pygame.time.get_ticks()
        if time_now - last_pipe > pipe_freq:
            pipe_height = random.randint(-160, +160)
            btm_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, -1)
            top_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, 1)
            pipe_group.add(btm_pipe)
            pipe_group.add(top_pipe)
            last_pipe = time_now

        # draw and scroll ground
        ground_scroll -= scroll_speed
        if abs(ground_scroll) > 35:
            ground_scroll = 0

        pipe_group.update()

    # check for game over and reset
    if game_over == True and flappy.rect.y > 0.9 * floor_location:
        if button.draw() == True:
            game_over = False
            score = reset_game()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN and flying == False and game_over == False:
            flying = True

    pygame.display.update()

pygame.quit()