from typing import Any
import pygame
from pygame.locals import *
import random
import numpy as np
import sys

from config import *
#from RL_main import gamma, reward_per_frame, J_star

#def get_di_dj_NN(NN, NN_input):
#    pass

gamma = 0.95
reward_per_frame = 20
J_star = reward_per_frame / (1 - gamma)
actor_boundary = 0.5


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x,y))


class Bird(pygame.sprite.Sprite):
    
    def __init__(self, x, y, params, policy=None): #, critic):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.params = params
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

        self.flying = True
        self.game_over = False

        self.u = 0

        self.policy = policy
        #self.critic = critic
    def update_action(self, action):
        self.u = action

    #def update_state(self, state):
    #    self.state = state

    def update(self):

        gravity = self.params['gravity']
        k = self.params['k']
        floor_location = self.params['floor_location']
        
        if self.flying == True:
            # gravity
            self.vel += gravity - k * self.vel * np.abs(self.vel)
            #if self.vel > 8:  # duh, revise that
            #    self.vel = 8
            if self.rect.bottom < floor_location:
                self.rect.y += int(self.vel)

        if self.game_over == False:

            if self.policy == None:

                if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:  # 0-th index for left click
                    self.vel = -8.5  # -7.5
                    self.clicked = True

                if pygame.mouse.get_pressed()[0] == 0:  # 0-th index for left click
                    self.clicked = False

            else:

                if self.u == 1 and self.clicked == False:
                    self.vel = -8.5  # -7.5
                    self.clicked = True

                if self.u == 0:
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

    #@property
    #def vel(self):
    #    return self.vel

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position, game_params):
        pygame.sprite.Sprite.__init__(self)
        self.scroll_speed = game_params['scroll_speed']
        self.image = pygame.image.load('images/pipe.png')
        self.rect = self.image.get_rect()
        pipe_gap = game_params['pipe_gap']
        # position 1 is from the top, -1 is from the bottom
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap/2)]
        if position == -1:
            self.rect.topleft = [x, y + int(pipe_gap/2)]

    def update(self):
        self.rect.x -= self.scroll_speed
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
    
