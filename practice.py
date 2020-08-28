colors = [['R','G','G','R','R'],
            ['R','R','G','R','R'],
            ['R','R','G','G','R'],
            ['R','R','R','R','R']]

measurements = ['G','G','G','G','G']

motions = [[0,0],
            [0,1],
            [1,0],
            [1,0],
            [0,1]]

sensor_right = 0.7
p_move = 0.8

def q4_localize(colors, measurements, motions, sensor_right, p_move):

    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]

    for i in range(len(measurements)):
        p = move(p, motions[i])
        p = sense(p, measurements[i])

    for i in range(len(p)):
        print(p[i])

    return p


def sense(p, Z):

    for row in range(len(p)):

        for col in range(len(p[row])):

            cell = colors[row][col]

            if cell == Z:
                p[row][col] = p[row][col] * sensor_right
            else:
                p[row][col] = p[row][col] * (1 - sensor_right)

    # Now we need to normalise the above posterior prob
    sum_p = sum(sum(p, []))

    for row in range(len(p)):
        for col in range(len(p[0])):
            p[row][col] = p[row][col] / sum_p

    return p


def move(p, step):

    q = [[0.0 for row in range(len(p[0]))] for col in range(len(p))]

    for row in range(len(p)):

        for col in range(len(p[row])):
            q[row][col] = p_move * p[(row - step[0]) % len(p)][(col - step[1]) % len(p[row])] + p[row][col] * (
                        1 - p_move)

    return q

print(q4_localize(colors,measurements,motions,sensor_right,p_move))