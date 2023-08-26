from flappy_bird_functions import *

def run_game(game_params=game_params, policy=None, critic=None, actor=None):

    pygame.init()

    #load game variables

    ground_scroll = 0
    scroll_speed = game_params['scroll_speed']
    floor_location = game_params['floor_location']
    pipe_freq = game_params['pipe_freq']  # [ms]
    last_pipe = pygame.time.get_ticks() - pipe_freq

    bird_group = pygame.sprite.Group()
    pipe_group = pygame.sprite.Group()

    x0 = int(screen_width/6)
    y0 = int(screen_height/2)
    flappy = Bird(x0, y0, game_params, policy)# , critic)  # initial position of bird
    #flappy.update_state([y0, 0, screen_width, y0])
    

    bird_group.add(flappy)

    score = 0
    cum_reward = 0
    rewards = np.empty(1000, dtype=float)
    J_values = np.empty(1000, dtype=float)
    all_critic_inputs = [None] * 1000
    step_idx = 0
    pass_pipe = False
    run = True

    first_pipe_created = False
    critic_input = [0.0] * 8

    u_prev = 0

    while run:

        clock.tick(fps)

        # draw background
        screen.blit(bg, (0,-100))   # image drawn from top-left corner
        bird_group.draw(screen)

        if first_pipe_created == True:

            all_pipes = pipe_group.sprites()
            n_btm_pipes = int(len(all_pipes)/2)
            btm_pipes = [None] * n_btm_pipes
            for i in range(n_btm_pipes):
                btm_pipes[i] = all_pipes[2*i]
            
            #state = [flappy.vel, flappy.rect.y, btm_pipe.rect.x, btm_pipe.rect.y]
            state = [flappy.vel, flappy.rect.y, 0.0, 0.0, 0.0, 0.0, u_prev]
            for i in range(n_btm_pipes):
                state[2 + 2*i] = btm_pipes[i].rect.x
                state[2 + 2*i+1] = btm_pipes[i].rect.y
            
            u = policy(actor, state)
            actor_output = actor.evaluate(state)  # to print
            critic_input[:7] = state
            critic_input[-1] = u
            critic_output = critic.evaluate(critic_input)
            J_val = critic_output[0]
            J_values[step_idx] = J_val

            flappy.update_action(u)

            cum_reward = cum_reward + reward_per_frame
            rewards[step_idx] = reward_per_frame
            step_idx = step_idx + 1

            if step_idx >0:

                dJ_du = critic.get_di_dj(critic_input, 0, 7)   # dJ/du: derivative of 0th component of critic NN output wrt to its 6th input component
                ea = J_val - J_star
                Ea = 0.5 * ea**2
                actor.backprop_2([dJ_du * ea])
                actor.apply_and_reset_gradients(1)
                
                ec = J_values[step_idx-1] - (gamma * J_values[step_idx] - rewards[step_idx])
                ec = 0 if ec < 1e-30 else ec
                Ec = 0.5 * ec**2
                critic.backprop_2([gamma*ec])
                critic.apply_and_reset_gradients(1)

                print(f"t={step_idx}; actor_output={actor_output} u={u}, J={J_val} ea={ea}, ec={ec}, dJ/du={dJ_du}")


            u_prev = u
                

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
                    #cum_reward = cum_reward + 100
                    #rewards[step_idx] = 10

        draw_text(str(score), font, white, int(screen_width/2), int(0.1*screen_height))

        # check for collision
        if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0: # "due kill args": kill bird and pipe
            flappy.game_over = True
            #pygame.quit()
            break

        # check if bird has hit ground
        if flappy.rect.bottom > floor_location:
            flappy.game_over = True
            flappy.flying = False
            #pygame.quit()
            break

        if flappy.game_over == False and flappy.flying == True:

            # generate new pipes
            time_now = pygame.time.get_ticks()
            if time_now - last_pipe > pipe_freq:
                pipe_height = random.randint(-160, +160)
                btm_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, -1, game_params)
                first_pipe_created = True
                top_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, 1, game_params)
                pipe_group.add(btm_pipe)
                pipe_group.add(top_pipe)
                last_pipe = time_now

            # draw and scroll ground
            ground_scroll -= scroll_speed
            if abs(ground_scroll) > 35:
                ground_scroll = 0

            pipe_group.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            #if event.type == pygame.MOUSEBUTTONDOWN and flying == False and game_over == False:
            #    flying = True

        pygame.display.update()

    critic_updated = critic
    actor_updated = actor

    return cum_reward, rewards[:step_idx], J_values[:step_idx], critic_updated, actor_updated

    #pygame.quit()
