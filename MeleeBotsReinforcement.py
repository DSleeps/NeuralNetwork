#!/usr/bin/python3
import melee
from melee import enums
import argparse
import signal
import sys
import torch
import numpy as np
import PyTorchNet as NN
import random as r

#This example program demonstrates how to use the Melee API to run dolphin programatically,
#   setup controllers, and send button presses over to dolphin

port1 = 1
port2 = 2

max_x = 165.0
max_y = 150.0

max_hitbox_size = 10.0
max_speed = 10.0

current_frame = 0

#How frequently the bot inputs something
frame_frequency = 3

#The rewards at the specific time step
rewards1 = []
rewards2 = []
choices1 = []
choices2 = []
current_count = 0

#The random chance to pick a different input
random_chance = 0.05

max_hitlag_frames = 60
max_hitstun_frames = 60

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
         raise argparse.ArgumentTypeError("%s is an invalid controller port. \
         Must be 1, 2, 3, or 4." % value)
    return ivalue

chain = None

parser = argparse.ArgumentParser(description='Example of libmelee in action')

parser.add_argument('--port', '-p', type=check_port,
                    help='The controller port your AI will play on',
                    default=2)
parser.add_argument('--opponent', '-o', type=check_port,
                    help='The controller port the opponent will play on',
                    default=3)
parser.add_argument('--live', '-l',
                    help='The opponent is playing live with a GCN Adapter',
                    default=False)
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game state')
parser.add_argument('--framerecord', '-r', default=False, action='store_true',
                    help='(DEVELOPMENT ONLY) Records frame data from the match, stores into framedata.csv.')

args = parser.parse_args()
print(args)

log = None
if args.debug:
    log = melee.logger.Logger()

framedata = melee.framedata.FrameData(args.framerecord)

#Options here are:
#   "Standard" input is what dolphin calls the type of input that we use
#       for named pipe (bot) input
#   GCN_ADAPTER will use your WiiU adapter for live human-controlled play
#   UNPLUGGED is pretty obvious what it means
opponent_type = melee.enums.ControllerType.UNPLUGGED
if args.live:
    opponent_type = melee.enums.ControllerType.GCN_ADAPTER

#Create our Dolphin object. This will be the primary object that we will interface with
dolphin = melee.dolphin.Dolphin(ai_port=port1,
                                opponent_port=args.opponent,
                                opponent_type=opponent_type,
                                logger=log)

#Create our GameState object for the dolphin instance
gamestate = melee.gamestate.GameState(dolphin)
#Create our Controller object that we can press buttons on
controller1 = melee.controller.Controller(port=port1, dolphin=dolphin)
controller2 = melee.controller.Controller(port=port2, dolphin=dolphin)

def signal_handler(signal, frame):
    dolphin.terminate()
    if args.debug:
        log.writelog()
        print("") #because the ^C will be on the terminal
        print("Log file created: " + log.filename)
    print("Shutting down cleanly...")
    if args.framerecord:
        framedata.saverecording()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#Run dolphin and render the output
#dolphin.run(render=True)
print('Here')

#Plug our controller1 in
#   Due to how named pipes work, this has to come AFTER running dolphin
#   NOTE: If you're loading a movie file, don't connect the controller1,
#   dolphin will hang waiting for input and never receive it

#---------------------My Logic------------------------#

#Here's where we create all of the starting networks
network_num = 2
networks = []
settings = {"layer_num": 8, "hidden_num": 100, "output_num": 45, "input_num": 23}

for i in range(network_num):
    networks.append(NN.NeuralNet(settings))

controller1.connect()
print('made it')
controller2.connect()
print('hereee')

state = "fight"

p_stock1 = 99
p_stock2 = 99

p_percent1 = 0
p_percent2 = 0

p_lost = -1

#The current neural networks
bot1 = 0
bot2 = 1

n1 = networks[bot1]
n2 = networks[bot2]

#Main loop
while True:
    #"step" to the next frame
    gamestate.step()
    if(gamestate.processingtime * 1000 > 12):
        print("WARNING: Last frame took " + str(gamestate.processingtime*1000) + "ms to process.")

    #What menu are we in?
    if gamestate.menu_state in [melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
        if args.framerecord:
            framedata.recordframe(gamestate)
        #XXX: This is where your AI does all of its stuff!
        #This line will get hit once per frame, so here is where you read
        #   in the gamestate and decide what buttons to push on the controller1

        players = gamestate.player
        p1 = players[port1]
        p2 = players[port2]

        me = players[4]

        current_frame += 1
        if (current_frame % frame_frequency == 0):
            current_frame = 0
        if (state == "fight" and current_frame % frame_frequency == 0):
            if (p1.percent > p_percent1):
                n2.add_reward(p1.percent-p_percent1)
                n1.add_reward(-(p1.percent-p_percent1))
            if (p2.percent > p_percent2):
                n1.add_reward(p2.percent-p_percent2)
                n2.add_reward(-(p2.percent-p_percent2))

            #If there percent goes down it means they die
            if (p1.action == enums.Action.ON_HALO_DESCENT):
                p_lost = 1
                n1.add_reward(-50)
                print('Blehh1')
                controller1.simple_press(0.5, 0.5, None)
                controller2.simple_press(0.5, 0.5, None)
            elif (p2.action == enums.Action.ON_HALO_DESCENT):
                p_lost = 2
                n2.add_reward(-50)
                print('Blehh')
                controller1.simple_press(0.5, 0.5, None)
                controller2.simple_press(0.5, 0.5, None)
            else:
                n1_inputs = np.zeros((settings["input_num"], 1))
                n1_inputs[0][0] = p1.x/max_x
                n1_inputs[1][0] = p1.y/max_y
                n1_inputs[2][0] = p2.x/max_x
                n1_inputs[3][0] = p2.y/max_y
                n1_inputs[4][0] = (p1.x - p2.x)/max_x
                n1_inputs[5][0] = (p1.y - p2.y)/max_y
                n1_inputs[6][0] = p1.speed_air_x_self / max_speed
                n1_inputs[7][0] = p1.speed_y_self / max_speed
                n1_inputs[8][0] = p1.speed_ground_x_self / max_speed
                n1_inputs[9][0] = p2.speed_air_x_self / max_speed
                n1_inputs[10][0] = p2.speed_y_self / max_speed
                n1_inputs[11][0] = p2.speed_ground_x_self / max_speed
                n1_inputs[12][0] = float(p2.hitbox_1_size) * int(p2.hitbox_1_status) / max_hitbox_size
                n1_inputs[13][0] = float(p2.hitbox_2_size) * int(p2.hitbox_2_status) / max_hitbox_size
                n1_inputs[14][0] = float(p2.hitbox_3_size) * int(p2.hitbox_3_status) / max_hitbox_size
                n1_inputs[15][0] = float(p2.hitbox_4_size) * int(p2.hitbox_4_status) / max_hitbox_size
                n1_inputs[16][0] = p1.hitlag_frames_left / max_hitlag_frames
                n1_inputs[17][0] = p1.hitstun_frames_left / max_hitstun_frames
                n1_inputs[18][0] = p1.jumps_left
                n1_inputs[19][0] = int(p1.on_ground)*1.0
                n1_inputs[20][0] = p1.action_frame / 60.0
                n1_inputs[21][0] = p1.percent / 150.0
                n1_inputs[22][0] = p2.percent / 150.0

                n2_inputs = np.zeros((settings["input_num"], 1))
                n2_inputs[0][0] = p2.x/max_x
                n2_inputs[1][0] = p2.y/max_y
                n2_inputs[2][0] = p1.x/max_x
                n2_inputs[3][0] = p1.y/max_y
                n2_inputs[4][0] = (p2.x - p1.x)/max_x
                n2_inputs[5][0] = (p2.y - p1.y)/max_y
                n2_inputs[6][0] = p2.speed_air_x_self / max_speed
                n2_inputs[7][0] = p2.speed_y_self / max_speed
                n2_inputs[8][0] = p2.speed_ground_x_self / max_speed
                n2_inputs[9][0] = p1.speed_air_x_self / max_speed
                n2_inputs[10][0] = p1.speed_y_self / max_speed
                n2_inputs[11][0] = p1.speed_ground_x_self / max_speed
                n2_inputs[12][0] = float(p1.hitbox_1_size) * int(p1.hitbox_1_status) / max_hitbox_size
                n2_inputs[13][0] = float(p1.hitbox_2_size) * int(p1.hitbox_2_status) / max_hitbox_size
                n2_inputs[14][0] = float(p1.hitbox_3_size) * int(p1.hitbox_3_status) / max_hitbox_size
                n2_inputs[15][0] = float(p1.hitbox_4_size) * int(p1.hitbox_4_status) / max_hitbox_size
                n2_inputs[16][0] = p2.hitlag_frames_left / max_hitlag_frames
                n2_inputs[17][0] = p2.hitstun_frames_left / max_hitstun_frames
                n2_inputs[18][0] = p2.jumps_left
                n2_inputs[19][0] = int(p2.on_ground)*1.0
                n2_inputs[20][0] = p2.action_frame / 60.0
                n2_inputs[21][0] = p2.percent / 150.0
                n2_inputs[22][0] = p1.percent / 150.0

                #Calculate the outputs
                n1_outputs = n1.forward_pass(torch.from_numpy(n1_inputs.T).float())
                n2_outputs = n2.forward_pass(torch.from_numpy(n2_inputs.T).float())
                control1 = n1_outputs[0]
                control2 = n2_outputs[0]

                #Tilt the controller
                if (control1 % 5 == 0):
                    tx1 = 0.5
                    ty1 = 0.5
                elif (control1 % 5 == 1):
                    tx1 = 1.0
                    ty1 = 0.5
                elif (control1 % 5 == 2):
                    tx1 = 0.5
                    ty1 = 1.0
                elif (control1 % 5 == 3):
                    tx1 = 0.0
                    ty1 = 0.5
                elif (control1 % 5 == 4):
                    tx1 = 0.5
                    ty1 = 0.0

                if (control2 % 5 == 0):
                    tx2 = 0.5
                    ty2 = 0.5
                elif (control2 % 5 == 1):
                    tx2 = 1.0
                    ty2 = 0.5
                elif (control2 % 5 == 2):
                    tx2 = 0.5
                    ty2 = 1.0
                elif (control2 % 5 == 3):
                    tx2 = 0.0
                    ty2 = 0.5
                elif (control2 % 5 == 4):
                    tx2 = 0.5
                    ty2 = 0.0

                #Now figure out what button to press
                control1 = int(control1/5)
                control2 = int(control2/5)

                if (control1 % 9 == 0):
                    pass
                elif (control1 % 9 == 1):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_A)
                elif (control1 % 9 == 2):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_B)
                elif (control1 % 9 == 3):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_X)
                elif (control1 % 9 == 4):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_Y)
                elif (control1 % 9 == 5):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_X)
                elif (control1 % 9 == 6):
                    controller1.simple_press(tx1, ty1, enums.Button.BUTTON_Z)
                elif (control1 % 9 == 7):
                    controller1.press_shoulder(enums.Button.BUTTON_L, 1)
                    controller1.press_shoulder(enums.Button.BUTTON_R, 0)
                elif (control1 % 9 == 8):
                    controller1.press_shoulder(enums.Button.BUTTON_R, 1)
                    controller1.press_shoulder(enums.Button.BUTTON_L, 0)

                if (control2 % 9 == 0):
                    pass
                elif (control2 % 9 == 1):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_A)
                elif (control2 % 9 == 2):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_B)
                elif (control2 % 9 == 3):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_X)
                elif (control2 % 9 == 4):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_Y)
                elif (control2 % 9 == 5):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_X)
                elif (control2 % 9 == 6):
                    controller2.simple_press(tx2, ty2, enums.Button.BUTTON_Z)
                elif (control2 % 9 == 7):
                    controller2.press_shoulder(enums.Button.BUTTON_L, 1)
                    controller2.press_shoulder(enums.Button.BUTTON_R, 0)
                elif (control2 % 9 == 8):
                    controller2.press_shoulder(enums.Button.BUTTON_R, 1)
                    controller2.press_shoulder(enums.Button.BUTTON_L, 0)

                for i in range(0):
                    pass

                current_count += 1
                if (current_count == 2*n1.batch_size):
                    n1.back_pass()
                    n2.back_pass()
                    current_count = n1.batch_size

        if (state != "reset"):
            p_stock1 = p1.stock
            p_stock2 = p2.stock
            p_percent1 = p1.percent
            p_percent2 = p2.percent
        # if args.framerecord:
        #     melee.techskill.upsmashes(ai_state=gamestate.ai_state, controller=controller1)
        # else:
        #     melee.techskill.multishine(ai_state=gamestate.ai_state, controller=controller1)
    #If we're at the character select screen, choose our character
    elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
        #Reset all of the things
        p_stock1 = 99
        p_stock2 = 99

        p_percent1 = 0
        p_percent2 = 0

        p_lost = -1

        #The current neural networks
        bot1 = 0
        bot2 = 1
        melee.menuhelper.choosecharacter(character=melee.enums.Character.FOX,
                                        gamestate=gamestate,
                                        port=port1,
                                        opponent_port=args.opponent,
                                        controller=controller1,
                                        swag=True,
                                        start=True)
        melee.menuhelper.choosecharacter(character=melee.enums.Character.FOX,
                                        gamestate=gamestate,
                                        port=port2,
                                        opponent_port=args.opponent,
                                        controller=controller2,
                                        swag=True,
                                        start=True)
    #If we're at the postgame scores screen, spam START
    elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
        # melee.menuhelper.skippostgame(controller=controller1)
        # melee.menuhelper.skippostgame(controller=controller2)
        controller1.simple_press(0.5, 0.5, enums.Button.BUTTON_START)
        controller2.simple_press(0.5, 0.5, enums.Button.BUTTON_START)
    #If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
        melee.menuhelper.choosestage(stage=melee.enums.Stage.BATTLEFIELD,
                                    gamestate=gamestate,
                                    controller=controller1)
    #Flush any button presses queued up
    controller1.flush()
    controller2.flush()
    if log:
        log.logframe(gamestate)
        log.writeframe()
