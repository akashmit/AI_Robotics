
from builtins import object
import random
from matrix import matrix
import math


class Pilot(object):
    timer_counter = 0

    def __init__(self, min_dist, in_bounds):
        self.min_dist = min_dist
        self.in_bounds = in_bounds
        self.dic = {}  # This dic contains information of ID: (X, P) for each asteroid

    def observe_asteroids(self, asteroid_locations):
        """ self - pointer to the current object.
           asteroid_locations - a list of asteroid observations. Each 
           observation is a tuple (i,x,y) where i is the unique ID for
           an asteroid, and x,y are the x,y locations (with noise) of the
           current observation of that asteroid at this time-step.
           Only asteroids that are currently 'in-bounds' will appear
           in this list, so be sure to use the asteroid ID, and not
           the position/index within the list to identify specific
           asteroids. (The list may change in size as asteroids move
           out-of-bounds or new asteroids appear in-bounds.)

           Return Values:
                    None
        """

        # This is where Measurement happens, followed by estimation

        # H is [2x6]
        H = matrix(([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]]))

        # R is still [2x2]
        R = matrix(([[1, 0],
                    [0, 1]]))

        # I is [6x6]
        Ident_Matrix = matrix(([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]]))

        # F is [6x6]
        F = matrix(([[1, 0, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 1],
                     [0, 0, 1, 0, 1, 0],
                     [0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]))

        # We need to loop over all asteroids. Each asteroid has its own KF
        for j in range(len(asteroid_locations)):

            # These hold the asteroid ID, current X location, current Y location
            ast_id, ast_x, ast_y = asteroid_locations[j]

            # First we need to check if this ast_id exists. If yes, we can ahead with the measurement step
            if ast_id in self.dic:

                # We want to get the x and P matrices for this asteroid
                x = self.dic[ast_id][0]
                P = self.dic[ast_id][1]

                # We do Kalman Filtering here!

                z = matrix([[ast_x], [ast_y]])     # z is [2x1]
                y = z - (H * x)                         # y is [2x1]. H is (2x6). x is (6x1)
                # Step1- Find the Kalman Gain
                S = (H * P) * (H.transpose()) + R    # S is [2x2]. P is (6x6). R is [2x2]
                K = P * (H.transpose()) * (S.inverse())  # We re-calculate Kalman Gain each time. K is [6x2]

                x = x + K * y  # We re-calculate new state each time

                P = (Ident_Matrix - (K * H)) * P  # We update P each time.

                # Now we have measured at time t. At time t, we make a PREDICTION for (t+1)
                x = (F * x)

                P = F * P * F.transpose()

                # Now we store these new (x, P) in dic
                self.dic[ast_id] = (x, P)

            # If ast_id is not in dic, we need to initialise x and P MATRICES and store them in dic[ast_id]
            else:
                x = matrix([[ast_x],
                            [ast_y],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0]])
                P = matrix(([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 10, 0, 0, 0],
                             [0, 0, 0, 10, 0, 0],
                             [0, 0, 0, 0, 10, 0],
                             [0, 0, 0, 0, 0, 10]]))

                self.dic[ast_id] = (x, P)

    def estimate_asteroid_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """

        res = []
        for key, value in self.dic.items():

            ast_id = key
            ast_x = value[0][0]
            ast_x = ast_x[0]

            ast_y = value[0][1]
            ast_y = ast_y[0]
            s = (ast_id, ast_x, ast_y)

            res.append(s)

        return res

    def next_move(self, craft_state):
        """ self - a pointer to the current object.
            craft_state - implemented as CraftState in craft.py.

            return values: 
              angle change: the craft may turn left(1), right(-1), 
                            or go straight (0). 
                            Turns adjust the craft's heading by 
                             angle_increment.
              speed change: the craft may accelerate (1), decelerate (-1), or 
                            continue at its current velocity (0). Speed 
                            changes adjust the craft's velocity by 
                            speed_increment, maxing out at max_speed.
         """

        self.timer_counter += 1
        if self.timer_counter < 15:
            return 0, 0

        # Slow down the craft every 5 moves in order to avoid being too fast
        if self.timer_counter % 5 == 0:
            return 0, -1

        # Turn the craft right and go right a few steps to avoid cluster of asteroids in beginning
        if self.timer_counter < 20:
            return -1, 0

        if self.timer_counter < 23:
            return 0, 1

        # Main thing to do here is to determine the 9 possible future estimates of the craft (determined at t for t+1)
        # So get 9 pairs of (x, y). Out of these filter anf pick 1. Return the angle and speed for this location

        x_axis_min = self.in_bounds.x_bounds[0]
        x_axis_max = self.in_bounds.x_bounds[1]
        y_axis_min = self.in_bounds.y_bounds[0]
        y_axis_max = self.in_bounds.y_bounds[1]

        # In locations we store (id, x, y, i, j) where ID is location_id; i is angle; j is speed
        locations = {}

        counter = 0
        for i in range(-1, 2):
            for j in range(-1, 2):

                craft_location = craft_state.steer(i, j)

                if (craft_location.x < x_axis_min or craft_location.x > x_axis_max
                        or craft_location.y < y_axis_min or craft_location.y > y_axis_max):
                    continue

                s = (craft_location.x, craft_location.y, i, j)
                locations[counter + 1] = s

                counter += 1

        # For this we need to loop over each location and calc the Euc_distance from each asteroid

        invalid_locations = []

        if len(self.dic) > 150:
            euc_comparison = self.min_dist + 0.02
        else:
            euc_comparison = self.min_dist + 0.04

        # We loop over each location- 1:9
        for key1, value1 in locations.items():

            craft_id = key1
            craft_x = value1[0]
            craft_y = value1[1]

            # We loop over each asteroid. We get future estimates from self.dic
            for key, value in self.dic.items():

                ast_x = value[0][0]
                ast_x = ast_x[0]
                ast_y = value[0][1]
                ast_y = ast_y[0]

                # We then calculate the Euc distance below and if it is less than min distance, add to invalid locations
                euc_dis = math.dist((craft_x, craft_y), (ast_x, ast_y))

                if euc_dis < euc_comparison:
                    invalid_locations.append(craft_id)
                    break

        # Now we remove all invalid locations
        for invalid_id in invalid_locations:
            locations.pop(invalid_id, None)

        # If head is North AND North loc: (6) is in locations we will go up (0, 1)
        # Else if head is right AND Left loc of right: (8) is in locations, we go left (1, 1)
        # Else we go Right (-1, 1)

        # If we can keep going up (0, 1), keep doing that.

        if 1.1 < craft_state.h < 2.4 and 6 in locations:
            return 0, 1
        elif 0 < craft_state.h < 1.7 and 3 in locations:
            return 1, 1
        elif craft_state.h < 1.6:
            return 1, 1

        return -1, 1
