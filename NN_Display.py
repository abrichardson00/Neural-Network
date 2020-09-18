import pygame

class NN_Display(object):

    """
    Class for displaying a neural network (from the NN class).
    Initializes a pygame window and contains seperate functions to draw evenly spaced neurons and the weights of the network.
    drawWeights() can be called repeatedly during network training to animate how the weights change.
    drawNeurons() must be called once (to calculate the neuron positions on the screen) before drawWeights() can be called.
    """

    def __init__(self,w,h):
        self.width = w
        self.height = h
        self.screen = pygame.display.set_mode((w,h))
        pygame.display.set_caption('Neural Network')
        self.screen.fill((0,0,0))
        pygame.display.flip()

        self.neuron_positions = None

    def updateDisplay(self):
        pygame.display.flip()

    def quitDisplay(self):
        pygame.quit()

    def drawNeurons(self,NN):
        self.neurons_per_layer = [NN.input_dimension] + NN.neurons_per_layer
        max_no_of_neurons = max(self.neurons_per_layer)
        x_distance_between_neurons = int(self.width / (len(self.neurons_per_layer) + 1))
        y_distance_between_neurons = int(self.height / (max_no_of_neurons + 1))
        ### get all positions of neurons
        self.neuron_positions = []
        neuron_x = 0
        for i in range(len(self.neurons_per_layer)):
            self.neuron_positions.append([])
            neuron_x += x_distance_between_neurons
            neuron_y = int((self.height / 2) - ( (self.neurons_per_layer[i]/2) - 0.5)*y_distance_between_neurons)
            for j in range(self.neurons_per_layer[i]):
                self.neuron_positions[i].append([neuron_x,neuron_y])
                if i == 0:
                    ### draw input layer as squares instead
                    pygame.draw.rect(self.screen, (255,255,255),[self.neuron_positions[i][j][0]-2,self.neuron_positions[i][j][1]-2,4,4])
                else:
                    pygame.draw.circle(self.screen, (255,255,255), self.neuron_positions[i][j], 3)
                neuron_y += y_distance_between_neurons
        ### draw neurons
        self.updateDisplay()

    def drawWeights(self,NN):
        for i in range(len(NN.neurons_per_layer)):

            for j in range((NN.neurons_per_layer[i])):
                for w in range(self.neurons_per_layer[i]): # <- the layer behind current NN.neurons_per_layer[i]
                    line_thickness = 2
                    colour_intensity = int(NN.layers[i][j].weights[w]*40)
                    if colour_intensity > 200:
                        colour_intensity = 200
                        line_thickness = 3
                    elif colour_intensity < -200:
                        colour_intensity = -200
                        line_thickness = 3
                    colour_intensity += 55
                    if colour_intensity < 0:
                        colour = (0,0,-colour_intensity)
                    else:
                        colour = (colour_intensity, 0, 0)

                    #print(colour_intensity)
                    line_start = [self.neuron_positions[i][w][0] + 4,self.neuron_positions[i][w][1] ]
                    #print(line_start)
                    line_end   = [self.neuron_positions[i+1][j][0] - 4,self.neuron_positions[i+1][j][1]]
                    pygame.draw.line(self.screen, colour,line_start,line_end,2)
        self.updateDisplay()

    #def drawEvaluation(self,NN, input):
    def holdDisplay(self):
        clock = pygame.time.Clock()
        done = False
        while not done:
            # This limits the while loop to a max of 10 times per second.
            # Leave this out and we will use all CPU we can.
            clock.tick(10)
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    done=True
        self.quitDisplay()
