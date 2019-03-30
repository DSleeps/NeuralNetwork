#!/usr/bin/python3
import melee
from melee import enums
import argparse
import signal
import sys
import NeuralNet as NN
import numpy as np

#This example program demonstrates how to use the Melee API to run dolphin programatically,
#   setup controllers, and send button presses over to dolphin

port1 = 1
port2 = 2

max_x = 165.0
max_y = 150.0

max_hitbox_size = 10.0
max_speed = 10.0

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
network_num = 10
networks = []
settings = {"weight_mult": 2, "layer_num": 5, "hidden_num": 40, "output_num": 13, "input_num": 16, "layer_funcs": [NN.sigmoid, NN.sigmoid, NN.sigmoid, NN.sigmoid, NN.sigmoid, NN.sigmoid]}

for i in range(network_num):
    networks.append(NN.NeuralNetwork(True, settings))

print(networks[1].calculate_outputs(np.random.rand(16,1)))
print(networks[1].calculate_outputs(np.random.rand(16,1)))
print(networks[1].calculate_outputs(np.random.rand(16,1)))

controller1.connect()
print('made it')
controller2.connect()
print('hereee')

state = "fight"

p_stock1 = 4
p_stock2 = 4

p_lost = -1

#The current neural networks
bot1 = 0
bot2 = 1

next_bot = 2
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
        print('here')
        if (state == "fight"):
            #If there percent goes down it means they die
            if (p_stock1 > p1.stock):
                state = "reset"
                p_lost = 1
            elif (p_stock2 > p2.stock):
                state = "reset"
                p_lost = 2
            else:
                #Otherwise parse the inputs and feed them into the neural networks
                n1 = networks[bot1]
                n2 = networks[bot2]

                n1_inputs = np.zeros((n1.input_num, 1))
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
                n1_inputs[12][0] = p2.hitbox_1_size * int(p2.hitbox_1_status) / max_hitbox_size
                n1_inputs[13][0] = p2.hitbox_2_size * int(p2.hitbox_2_status) / max_hitbox_size
                n1_inputs[14][0] = p2.hitbox_3_size * int(p2.hitbox_3_status) / max_hitbox_size
                n1_inputs[15][0] = p2.hitbox_4_size * int(p2.hitbox_4_status) / max_hitbox_size

                n2_inputs = np.zeros((n2.input_num, 1))
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
                n2_inputs[12][0] = p1.hitbox_1_size * int(p1.hitbox_1_status) / max_hitbox_size
                n2_inputs[13][0] = p1.hitbox_2_size * int(p1.hitbox_2_status) / max_hitbox_size
                n2_inputs[14][0] = p1.hitbox_3_size * int(p1.hitbox_3_status) / max_hitbox_size
                n2_inputs[15][0] = p1.hitbox_4_size * int(p1.hitbox_4_status) / max_hitbox_size

                #Calculate the outputs
                n1_outputs = n1.calculate_outputs(n1_inputs)
                n2_outputs = n2.calculate_outputs(n2_inputs)

                #Put the outputs into the controller
                if (n1_outputs[0][0] > 0.75): controller1.press_button(enums.Button.BUTTON_A)
                else: controller1.release_button(enums.Button.BUTTON_A)
                if (n1_outputs[1][0] > 0.75): controller1.press_button(enums.Button.BUTTON_B)
                else: controller1.release_button(enums.Button.BUTTON_B)
                if (n1_outputs[2][0] > 0.75): controller1.press_button(enums.Button.BUTTON_X)
                else: controller1.release_button(enums.Button.BUTTON_X)
                if (n1_outputs[3][0] > 0.75): controller1.press_button(enums.Button.BUTTON_Y)
                else: controller1.release_button(enums.Button.BUTTON_Y)
                if (n1_outputs[4][0] > 0.75): controller1.press_button(enums.Button.BUTTON_Z)
                else: controller1.release_button(enums.Button.BUTTON_Z)
                if (n1_outputs[5][0] > 0.75): controller1.press_shoulder(enums.Button.BUTTON_L, n1_outputs[11][0])
                else: controller1.press_shoulder(enums.Button.BUTTON_L, 0)
                if (n1_outputs[6][0] > 0.75): controller1.press_shoulder(enums.Button.BUTTON_R, n1_outputs[12][0])
                else: controller1.press_shoulder(enums.Button.BUTTON_R, 0)

                xm1, ym1, xc1, yc1 = 0.5, 0.5, 0.5, 0.5
                if (n1_outputs[7][0] > 0.7): xm1 = 1
                elif (n1_outputs[7][0] < 0.3): xm1 = 0
                if (n1_outputs[8][0] > 0.7): ym1 = 1
                elif (n1_outputs[8][0] < 0.3): ym1 = 0
                if (n1_outputs[9][0] > 0.7): xc1 = 1
                elif (n1_outputs[9][0] < 0.3): xc1 = 0
                if (n1_outputs[10][0] > 0.7): yc1 = 1
                elif (n1_outputs[10][0] < 0.3): yc1 = 0
                controller1.tilt_analog(enums.Button.BUTTON_MAIN, xm1, ym1)
                controller1.tilt_analog(enums.Button.BUTTON_C, xc1, yc1)

                if (n2_outputs[0][0] > 0.75): controller2.press_button(enums.Button.BUTTON_A)
                else: controller2.release_button(enums.Button.BUTTON_A)
                if (n2_outputs[1][0] > 0.75): controller2.press_button(enums.Button.BUTTON_B)
                else: controller2.release_button(enums.Button.BUTTON_B)
                if (n2_outputs[2][0] > 0.75): controller2.press_button(enums.Button.BUTTON_X)
                else: controller2.release_button(enums.Button.BUTTON_X)
                if (n2_outputs[3][0] > 0.75): controller2.press_button(enums.Button.BUTTON_Y)
                else: controller2.release_button(enums.Button.BUTTON_Y)
                if (n2_outputs[4][0] > 0.75): controller2.press_button(enums.Button.BUTTON_Z)
                else: controller2.release_button(enums.Button.BUTTON_Z)
                if (n2_outputs[5][0] > 0.75): controller2.press_shoulder(enums.Button.BUTTON_L, n2_outputs[11][0])
                else: controller2.press_shoulder(enums.Button.BUTTON_L, 0)
                if (n2_outputs[6][0] > 0.75): controller2.press_shoulder(enums.Button.BUTTON_R, n2_outputs[12][0])
                else: controller2.press_shoulder(enums.Button.BUTTON_R, 0)

                xm2, ym2, xc2, yc2 = 0.5, 0.5, 0.5, 0.5
                if (n2_outputs[7][0] > 0.7): xm2 = 1
                elif (n2_outputs[7][0] < 0.3): xm2 = 0
                if (n2_outputs[8][0] > 0.7): ym2 = 1
                elif (n2_outputs[8][0] < 0.3): ym2 = 0
                if (n2_outputs[9][0] > 0.7): xc2 = 1
                elif (n2_outputs[9][0] < 0.3): xc2 = 0
                if (n2_outputs[10][0] > 0.7): yc2 = 1
                elif (n2_outputs[10][0] < 0.3): yc2 = 0
                controller2.tilt_analog(enums.Button.BUTTON_MAIN, xm2, ym2)
                controller2.tilt_analog(enums.Button.BUTTON_C, xc2, yc2)

        elif (state == "reset"):
            print('Switching')
            #If player 1 lost, player 2 must kill himself to reset the percent
            if (p_lost == 1):
                controller2.current.main_stick = (1.0, 0.5)
                if (p_stock1 > p1.stock):
                    state = "fight"
                    bot2 = next_bot
                    next_bot += 1
            #If the other player lost, do the opposite
            if (p_lost == 2):
                controller1.current.main_stick = (1.0, 0.5)
                if (p_stock2 > p2.stock):
                    state = "fight"
                    bot1 = next_bot
                    next_bot += 1

        print('X: ' + str(me.x) + ' Y: ' + str(me.y))

        p_stock1 = p1.stock
        p_stock2 = p2.stock
        # if args.framerecord:
        #     melee.techskill.upsmashes(ai_state=gamestate.ai_state, controller=controller1)
        # else:
        #     melee.techskill.multishine(ai_state=gamestate.ai_state, controller=controller1)
    #If we're at the character select screen, choose our character
    elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
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
        melee.menuhelper.skippostgame(controller=controller1)
    #If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
        melee.menuhelper.choosestage(stage=melee.enums.Stage.POKEMON_STADIUM,
                                    gamestate=gamestate,
                                    controller=controller1)
    #Flush any button presses queued up
    controller1.flush()
    controller2.flush()
    if log:
        log.logframe(gamestate)
        log.writeframe()
