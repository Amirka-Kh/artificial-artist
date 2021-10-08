import numpy as np
import cv2
from random import shuffle


# sets dot pic to an image
def set_dot(n_image):
    for i in range(5):
        # chooses a dot position
        c_x = np.random.randint(0, img_f[1])
        c_y = np.random.randint(0, img_f[0])

        # sets a radius (since it is a dot, the range is between 0 and 5.12)
        radius = np.random.randint(0, int(img_f[0] / 100.))

        # sets characteristics
        # opacity adds smoothness to an image (akvarell painting effect)
        opacity = np.random.rand(1)[0]
        color = set_rgb_color()

        # brings blank image
        overlay = n_image.copy()

        # draws small circle 'dot' on the overlay
        cv2.circle(overlay, (c_x, c_y), radius, color, -1)

        # adds this circle 'dot' to the image
        # this procedure is done to apply an opacity feature
        cv2.addWeighted(overlay, opacity, n_image, 1 - opacity, 0, n_image)


# sets circle pic to an image
def set_circle(n_image):
    # initializes a circle
    center_x = np.random.randint(0, img_f[1])
    center_y = np.random.randint(0, img_f[0])
    radius = np.random.randint(0, int(img_f[0] / (1.1 * res)))
    opacity = np.random.rand(1)[0]
    color = set_rgb_color()

    # draws the circle on an image
    overlay = n_image.copy()
    cv2.circle(overlay, (center_x, center_y), radius, color, -1)
    cv2.addWeighted(overlay, opacity, n_image, 1 - opacity, 0, n_image)


# sets rectangular to an image
def set_rect(n_image):
    # initializes positions
    min_x = np.random.randint(0, img_f[1])
    max_x = min_x + np.random.randint(5, int(img_f[1] / res))
    min_y = np.random.randint(0, img_f[0])
    max_y = min_y + np.random.randint(5, int(img_f[0] / res))

    # sets characteristics
    opacity = np.random.rand(1)[0]
    color = set_rgb_color()

    # draws rectangle on an image
    overlay = n_image.copy()
    cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1)
    cv2.addWeighted(overlay, opacity, n_image, 1 - opacity, 0, n_image)


# creates an individual\istance\random_image
def create_rimage():
    # creates a blank image
    n_image = np.zeros((img_f[0], img_f[1], 3), np.uint8)

    # adds random figures to the image
    for i in range(nmb_of_figs):
        fig_fdic[fig](n_image)
    return n_image


# sets a color
def set_rgb_color():
    # color in rgb, e.g. blue = (0, 0, 255)
    blue = np.random.randint(0, 255)
    green = np.random.randint(0, 255)
    red = np.random.randint(0, 255)
    return (blue, green, red)


# creates sm number of istances\indiciduals
def create_population():
    for i in range(popu_size):
        instance = create_rimage()

        # adds instance\individual to the population
        img_list.append(instance)


# evolutionary algorithm
def algorithm():
    # creates initial population from which evolution will start
    create_population()
    global img_list

    # runs evolution until 250000th generation
    for i in range(250000):
        if i % 1000 == 0:
            print("#%d th generation, best fitness: %.4f" % (i + 1, fitness[0]), end="\r")

        # initializes fitnesses and percentages for new generation\population
        set_fitness()
        set_percent()

        # sets new population
        new_pop = []
        new_pop.append(best_ind)
        new_pop.append(second_ind)

        # creates descendents
        for j in range(popu_size - 2):

            # selects ancestors
            x = random_sel()
            y = random_sel()

            child = born_by(x, y)

            # checks if mutation applicable (it should not happen very often)
            if (np.random.randint(10) < mut_rate):
                mutate(child)

            new_pop.append(child)

        # safes new population in population list
        img_list = new_pop.copy()

    show_best_image()


# selects ancestor
def random_sel():
    # to be calm, and do not wait suprises
    if len(percent_pool) == 0:
        print("percent pool size is 0")
        return img_list[np.random.randint(0, popu_size - 1)]

    # with high probability returns best ancestor
    return percent_pool[np.random.randint(0, len(percent_pool) - 1)]


# reproduces descendent from ancestors  (crossover)
def born_by(x, y):
    # sets percentage of how strong the influence will be
    # for the first ancestor
    x_weight = np.random.rand(1)[0]

    # creates blank images
    ne_image = np.zeros((img_f[0], img_f[1], 3), np.uint8)

    # overlays ancestors on each other
    # creates a child
    cv2.addWeighted(x, x_weight, y, 1 - x_weight, 0, ne_image)
    return ne_image


# mutates child
def mutate(x):
    # adds new figures to the picture\image\istance
    for i in range(np.random.randint(1, 3)):
        fig_fdic[fig](x)


# sets fitness
def set_fitness():
    global best_ind
    best_ind = img_list[0]
    global second_ind
    second_ind = img_list[1]

    for i in range(popu_size):
        # returns standard deviation
        distance = get_distance(i)

        # fitness will show how similar istance\individual to original image
        fitness[i] = (1. / distance)

        # defines best istances\indiciduals
        if fitness[i] > fitness[0]:
            best_ind = img_list[i]
        elif fitness[i] > fitness[1]:
            second_ind = img_list[1]


# returns standard deviation
def get_distance(i):
    # np.linalg.norm helps count sqrt of a sum of all (original.pix - istance.pix)**2
    # it is needed to find standard deviation, which is computed below
    return (np.linalg.norm(image.astype('float') - img_list[i].astype('float'))) / (img_f[0] * img_f[1])


# sets percent
def set_percent():
    total = 0
    for i in range(popu_size):
        total += fitness[i]

    for i in range(popu_size):
        # it will be used for selection (create_percent_pool)
        if fitness[i] is not 0 and total is not 0:
            percent[i] = int(1 + fitness[i] * 100 / total)
        # bad fitness results in low chances for being a parent
        else:
            percent[i] = 1

    create_percent_pool()


# creates pool where good descendents\istances appear othen
def create_percent_pool():
    percent_pool.clear()
    for i in range(popu_size):
        for j in range(percent[i]):
            percent_pool.append(img_list[i])
    shuffle(percent_pool)


def show_best_image():
    cv2.imshow("result", img_list[0])
    cv2.waitKey(0)
    cv2.imwrite('output.jpg', img_list[0])


""" variables to be changed for personal benefit, evolution settings """
# 'nmb_of_figs' number of figures in a new image
# 'popu_size' population size
# 'mut_rate' "how often"  algorithm will mutate children
# 'fig' stands for figure

nmb_of_figs = 50
popu_size = 50
mut_rate = 3
fig = 'circle'
img_path = 'baboon.jpg'


"""  variables, list and smth which we will need in our evolution """
# 'img_f' will help to adjust the height and width of the new image
# 'fig_fdic' dictionary to call functions, e. g. 'line' will bring 'set_line' function
# 'res' helps with figures location setting
# 'fitness' each instance\individual will carry a personal fitness value
# 'percent' same for the percent
# 'percent_pool' to make selection
# 'img_list' it is list where we will keep a population
# 'best_ind' and 'second_ind' stands for 2 best instances\individuals
img_f = [512, 512]
fig_fdic = {'circle': set_circle, 'rectangular': set_rect, 'dot': set_dot}
res = 10
fitness = [0] * popu_size
percent = [0] * popu_size
percent_pool = []
img_list = []
best_ind = None
second_ind = None

# transform image, run an algorithm
image = cv2.imread(img_path)
algorithm()
