from flappy_bird_functions import *

gamma = 0.95
crash_reward = -5000
step_reward = 5

def sample_from_buffer(D_buffer_input, global_D_idx, m=30):
    D_buffer = copy.deepcopy(D_buffer_input[:(global_D_idx+1)])
    length = len(D_buffer)
    if length<m:
        m = length

    minibatch_indices = np.random.choice(length, size=m, replace=False).tolist()
    #minibatch = (np.array(D_buffer)[minibatch_indices])
    minibatch = [D_buffer[i] for i in minibatch_indices]

    return minibatch, minibatch_indices

def get_yj_from_minibatch(minibatch_input, Q_hat):

    minibatch = copy.deepcopy(minibatch_input)
    
    n = len(minibatch)
    yj = [None] * n
    for i, D_vect in enumerate(minibatch):
        reward = D_vect[2]
        if reward == crash_reward:
            yj[i] = crash_reward
        else:
            state_next = D_vect[3]
            state_next.append(0.0)
            q0 = Q_hat.evaluate(state_next)
            state_next[-1] = 1.0
            q1 = Q_hat.evaluate(state_next)
            yj[i] = reward + gamma*np.max([q0, q1])

    return yj

def get_errors(yj_array, minibatch, Q_nn):
    assert len(yj_array) == len(minibatch)

    n = len(yj_array)
    differences = [None] * n
    estimates_array = [None] * n
    for i, D_vect in enumerate(minibatch):
        state = D_vect[0]
        action = D_vect[1]
        state.append(action)
        estimates_array[i] = Q_nn.evaluate(state)
        differences[i] = yj_array[i] - estimates_array[i]

    return differences, estimates_array

def normalise(state_input):
    factors = [v_terminal, screen_height, screen_width, screen_height, screen_width, screen_height, 1]
    state = [a/b for a, b in zip(state_input, factors)]
    return state



def run_game(take_action, Q_nn, Q_hat, D_buffer, global_D_idx, game_idx, game_params=game_params):

    pygame.init()

    ground_scroll = 0
    
    current_frame = 0
    scroll_speed = game_params['scroll_speed']
    floor_location = game_params['floor_location']
    pipe_freq = game_params['pipe_freq']  # [ms]
    last_pipe = pygame.time.get_ticks() - pipe_freq
    pipe_freq_frames = pipe_freq * 1e-3 * fps
    last_pipe_frame = current_frame - pipe_freq_frames

    bird_group = pygame.sprite.Group()
    pipe_group = pygame.sprite.Group()

    x0 = int(screen_width/6)
    y0 = int(screen_height/2)
    flappy = Bird(x0, y0, game_params, take_action)# , critic)  # initial position of bird
    #flappy.update_state([y0, 0, screen_width, y0])
    

    bird_group.add(flappy)

    score = 0
    cum_reward = 0
    rewards = np.empty(1000, dtype=float)
    step_idx = 0
    pass_pipe = False
    run = True

    first_pipe_created = False
    action_prev = 0

    while run:
        

        clock.tick(fps)
        current_frame += 1

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
            state = [flappy.vel, flappy.rect.centery, 0.0, 0.0, 0.0, 0.0]
            for i in range(n_btm_pipes):
                state[2 + 2*i] = btm_pipes[i].rect.x
                state[2 + 2*i+1] = btm_pipes[i].rect.y

            state.append(action_prev)

            state = normalise(state)

            if step_idx>1:
                D_vect_t = [state_prev, u, rt_prev, state]  # u is also from the previous instant

                #D_buffer.append(D_vect_t)
                D_buffer[global_D_idx] = D_vect_t

                minibatch, _ = sample_from_buffer(D_buffer, global_D_idx)

                #for D_vect in minibatch:
                #    if len(D_vect[0])>7 or len(D_vect[3])>7:
                #        input("Sths wrong")

                yj_array = get_yj_from_minibatch(minibatch, Q_hat)
                #differences, estimates = get_errors(yj_array, minibatch, Q_nn)
                input_states = [None] * len(minibatch) # [D_vect[0].append(D_vect[1]) for D_vect in minibatch]
                for i, D_vect in enumerate(copy.deepcopy(minibatch)):
                    input_states[i] = D_vect[0]
                    input_states[i].append(D_vect[1])

                #print(input_states)
                mse, std = Q_nn.train(input_states, yj_array, training_it=game_idx)
                #print(f"Error: {mse}")

                global_D_idx = global_D_idx + 1
                if global_D_idx == len(D_buffer):
                    D_buffer = [None] * global_D_idx
                    global_D_idx = 0
            
            u = take_action(Q_nn, state, game_idx)
            #print(f"u={u}")
            flappy.update_action(u)
        
            state_prev = copy.deepcopy(state)
            action_prev = copy.deepcopy(u)

        if global_D_idx % 10000 == 0:
            Q_hat = copy.deepcopy(Q_nn)

        bird_group.update()

        pipe_group.draw(screen)

        screen.blit(ground_img, (ground_scroll, floor_location))

        # check score
        next_pipe_y_idx = 3
        if len(pipe_group) > 0:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left\
                and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right\
                and pass_pipe == False:
                pass_pipe = True

            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                next_pipe_y_idx = 5
                if pass_pipe == True:
                    score += 1
                    pass_pipe = False
                    #cum_reward = cum_reward + 100
                    #rewards[step_idx] = 10

        draw_text(str(score), font, white, int(screen_width/2), int(0.1*screen_height))

        # check for collision
        if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0: # "due kill args": kill bird and pipe
            rt = crash_reward * np.abs(state[1] - (state[next_pipe_y_idx] - game_params['pipe_gap']/(2*screen_height)))
            cum_reward += rt
            D_vect_t = [state, u, rt, [0.0]*6]
            #print(f"Collision! Next pipe idx = {next_pipe_y_idx} - reward = {rt}")
            #D_buffer.append(D_vect_t)
            D_buffer[global_D_idx] = D_vect_t
            flappy.game_over = True
            #pygame.quit()
            break
            
        # we have not collided!
        step_idx = step_idx + 1
        #print(step_idx)
        if step_idx>1:
            rt_prev = step_reward / (np.abs(state[1] - (state[next_pipe_y_idx] - game_params['pipe_gap']/(2*screen_height))) + 1e-2)
            cum_reward += rt_prev

        # check if bird has hit ground
        if flappy.rect.bottom > floor_location:
            #D_vect_t = [state, u, crash_reward, [0.0]*6]
            rt = crash_reward * np.abs(state[1] - (state[next_pipe_y_idx] - game_params['pipe_gap']/(2*screen_height)))
            cum_reward += rt
            D_vect_t = [state, u, rt, [0.0]*6]
            #D_buffer.append(D_vect_t)
            D_buffer[global_D_idx] = D_vect_t
            flappy.game_over = True
            flappy.flying = False
            #pygame.quit()
            break

        if flappy.game_over == False and flappy.flying == True:

            # generate new pipes
            time_now = pygame.time.get_ticks()
            #if time_now - last_pipe > pipe_freq:
            if current_frame - last_pipe_frame > pipe_freq_frames:
                pipe_height = random.randint(-160, +160)
                btm_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, -1, game_params)
                first_pipe_created = True
                top_pipe = Pipe(int(screen_width), int(screen_height/2) + pipe_height, 1, game_params)
                pipe_group.add(btm_pipe)
                pipe_group.add(top_pipe)
                last_pipe = time_now
                last_pipe_frame = current_frame

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


    return Q_nn, Q_hat, D_buffer, global_D_idx, score, cum_reward

    #pygame.quit()
