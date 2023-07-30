import pygame
from utils import *
import numpy as np

class CarGameEnviroment:
    def __init__(self):
        pygame.init()


        # AI config
        self.num_actions = 4
        self.num_inputs = 12
        self.policy_loss = "None"
        

        # Game config
        self.screen_width = 1920 
        self.screen_height = 1080

        self.is_running = True

        self.font = pygame.font.Font('freesansbold.ttf', 32)
 
        # create a text surface object,
        # on which text is drawn on it.
        
        
        # create a rectangular object for the
        # text surface object
        self.text = self.font.render(str(self.policy_loss), True, (0, 255, 0), (0, 0, 255))
        self.textRect = self.text.get_rect()
        self.textRect.centerx = self.screen_width // 2
        self.textRect.centery =self.screen_height // 2 -120

        # When resizing adjust it
        self.goal_lines = [Line_Segment(Point(1131, 154), Point(1181, 374)), Line_Segment(Point(1319, 94), Point(1350, 369)), Line_Segment(Point(1488, 108), Point(1432, 361)), Line_Segment(Point(1782, 202), Point(1537, 394)), Line_Segment(Point(1570, 406), Point(1897, 446)), Line_Segment(Point(1852, 598), Point(1528, 626)), Line_Segment(Point(1492, 738), Point(1790, 864)), Line_Segment(Point(1432, 851), Point(1559, 1065)), Line_Segment(Point(1318, 849), Point(1230, 1052)), Line_Segment(Point(1064, 903), Point(1222, 657)), Line_Segment(Point(1012, 756), Point(959, 482)), Line_Segment(Point(668, 643), Point(795, 868)), Line_Segment(Point(565, 713), Point(506, 956)), Line_Segment(Point(351, 700), Point(212, 909)), Line_Segment(Point(313, 664), Point(32, 529)), Line_Segment(Point(479, 551), Point(163, 436)), Line_Segment(Point(429, 365), Point(146, 336)), Line_Segment(Point(433, 247), Point(176, 176))]
        self.borders = [Line_Segment(Point(1356, 301), Point(898, 401)), Line_Segment(Point(1622, 355), Point(1356, 301)), Line_Segment(Point(1672, 395), Point(1622, 355)), Line_Segment(Point(1581, 521), Point(1672, 395)), Line_Segment(Point(1549, 824), Point(1581, 521)), Line_Segment(Point(1430, 896), Point(1549, 824)), Line_Segment(Point(1283, 862), Point(1430, 896)), Line_Segment(Point(1176, 629), Point(1283, 862)), Line_Segment(Point(1040, 490), Point(1176, 629)), Line_Segment(Point(564, 760), Point(1040, 490)), Line_Segment(Point(319, 730), Point(564, 760)), Line_Segment(Point(231, 671), Point(319, 730)), Line_Segment(Point(355, 614), Point(231, 671)), Line_Segment(Point(443, 472), Point(355, 614)), Line_Segment(Point(356, 299), Point(443, 472)), Line_Segment(Point(398, 236), Point(356, 299)), Line_Segment(Point(514, 186), Point(398, 236)), Line_Segment(Point(630, 189), Point(514, 186)), Line_Segment(Point(724, 306), Point(630, 189)), Line_Segment(Point(898, 395), Point(724, 306)), Line_Segment(Point(1006, 376), Point(898, 395)), Line_Segment(Point(1352, 125), Point(923, 216)), Line_Segment(Point(1696, 190), Point(1352, 125)), Line_Segment(Point(1916, 360), Point(1696, 190)), Line_Segment(Point(1749, 586), Point(1916, 360)), Line_Segment(Point(1743, 913), Point(1749, 586)), Line_Segment(Point(1462, 1074), Point(1743, 913)), Line_Segment(Point(1160, 1011), Point(1462, 1074)), Line_Segment(Point(1006, 701), Point(1160, 1011)), Line_Segment(Point(620, 944), Point(1006, 701)), Line_Segment(Point(257, 901), Point(620, 944)), Line_Segment(Point(82, 792), Point(257, 901)), Line_Segment(Point(3, 586), Point(82, 792)), Line_Segment(Point(240, 468), Point(3, 586)), Line_Segment(Point(153, 294), Point(240, 468)), Line_Segment(Point(282, 100), Point(153, 294)), Line_Segment(Point(484, 6), Point(282, 100)), Line_Segment(Point(714, 17), Point(484, 6)), Line_Segment(Point(836, 172), Point(714, 17)), Line_Segment(Point(919, 211), Point(836, 172)), Line_Segment(Point(1066, 181), Point(919, 211))]

        
        # Car config
        self.car_width, self.car_height = 100, 50
        self.max_speed = 100  # Maximum speed of the car
        self.acceleration = 0.3  # Acceleration rate
        self.friction = 0.02
        
        self.turn_speed_base = 5
        self.turn_speed_adjust = 1  # Change this value to adjust the turning speed



        #Grafic config
        
        self.car_image = pygame.image.load("car.png")
        self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
        self.car_rect = self.car_image.get_rect()

        self.bg_image = pygame.image.load("Map2.png")
        self.bg_image = pygame.transform.scale(self.bg_image, (self.screen_width, self.screen_height))


        # Game init
        self.car_x = self.screen_width // 2
        self.car_y = 1080 // 8 + 100
        self.car_speed = 0  # Current car speed
        self.car_angle = 0
        
        
        


    
    def reset(self):
        self.car_x = self.screen_width // 2
        self.car_y = 1080 // 8 + 100
        self.car_speed = 0 
        self.car_angle = 0
        self.goal_lines = [Line_Segment(Point(1131, 154), Point(1181, 374)), Line_Segment(Point(1319, 94), Point(1350, 369)), Line_Segment(Point(1488, 108), Point(1432, 361)), Line_Segment(Point(1782, 202), Point(1537, 394)), Line_Segment(Point(1570, 406), Point(1897, 446)), Line_Segment(Point(1852, 598), Point(1528, 626)), Line_Segment(Point(1492, 738), Point(1790, 864)), Line_Segment(Point(1432, 851), Point(1559, 1065)), Line_Segment(Point(1318, 849), Point(1230, 1052)), Line_Segment(Point(1064, 903), Point(1222, 657)), Line_Segment(Point(1012, 756), Point(959, 482)), Line_Segment(Point(668, 643), Point(795, 868)), Line_Segment(Point(565, 713), Point(506, 956)), Line_Segment(Point(351, 700), Point(212, 909)), Line_Segment(Point(313, 664), Point(32, 529)), Line_Segment(Point(479, 551), Point(163, 436)), Line_Segment(Point(429, 365), Point(146, 336)), Line_Segment(Point(433, 247), Point(176, 176))]
        self.borders = [Line_Segment(Point(1356, 301), Point(898, 401)), Line_Segment(Point(1622, 355), Point(1356, 301)), Line_Segment(Point(1672, 395), Point(1622, 355)), Line_Segment(Point(1581, 521), Point(1672, 395)), Line_Segment(Point(1549, 824), Point(1581, 521)), Line_Segment(Point(1430, 896), Point(1549, 824)), Line_Segment(Point(1283, 862), Point(1430, 896)), Line_Segment(Point(1176, 629), Point(1283, 862)), Line_Segment(Point(1040, 490), Point(1176, 629)), Line_Segment(Point(564, 760), Point(1040, 490)), Line_Segment(Point(319, 730), Point(564, 760)), Line_Segment(Point(231, 671), Point(319, 730)), Line_Segment(Point(355, 614), Point(231, 671)), Line_Segment(Point(443, 472), Point(355, 614)), Line_Segment(Point(356, 299), Point(443, 472)), Line_Segment(Point(398, 236), Point(356, 299)), Line_Segment(Point(514, 186), Point(398, 236)), Line_Segment(Point(630, 189), Point(514, 186)), Line_Segment(Point(724, 306), Point(630, 189)), Line_Segment(Point(898, 395), Point(724, 306)), Line_Segment(Point(1006, 376), Point(898, 395)), Line_Segment(Point(1352, 125), Point(923, 216)), Line_Segment(Point(1696, 190), Point(1352, 125)), Line_Segment(Point(1916, 360), Point(1696, 190)), Line_Segment(Point(1749, 586), Point(1916, 360)), Line_Segment(Point(1743, 913), Point(1749, 586)), Line_Segment(Point(1462, 1074), Point(1743, 913)), Line_Segment(Point(1160, 1011), Point(1462, 1074)), Line_Segment(Point(1006, 701), Point(1160, 1011)), Line_Segment(Point(620, 944), Point(1006, 701)), Line_Segment(Point(257, 901), Point(620, 944)), Line_Segment(Point(82, 792), Point(257, 901)), Line_Segment(Point(3, 586), Point(82, 792)), Line_Segment(Point(240, 468), Point(3, 586)), Line_Segment(Point(153, 294), Point(240, 468)), Line_Segment(Point(282, 100), Point(153, 294)), Line_Segment(Point(484, 6), Point(282, 100)), Line_Segment(Point(714, 17), Point(484, 6)), Line_Segment(Point(836, 172), Point(714, 17)), Line_Segment(Point(919, 211), Point(836, 172)), Line_Segment(Point(1066, 181), Point(919, 211))]

        return np.ones(self.num_inputs)

    def step(self, action):
        reward = 0
        done = False
        
        # Go forward
        if action == 0:
            self.car_speed = min(self.car_speed + self.acceleration, self.max_speed)

        # Slow down (decelerate) or move backward
        if action == 1:
            self.car_speed = max(self.car_speed - self.acceleration, -self.max_speed)

        # Turn left
        if action == 2:
            self.car_angle = (self.car_angle + (self.turn_speed_base * self.turn_speed_adjust)) % 360

        # Turn right
        if action == 3:
            self.car_angle = (self.car_angle - (self.turn_speed_base * self.turn_speed_adjust)) % 360
        
        # apply friction
        self.car_speed *= (1 - self.friction)

        # Movenment
        car_accel_x = self.car_speed * math.cos(math.radians(self.car_angle))
        car_accel_y = -self.car_speed * math.sin(math.radians(self.car_angle))
        self.car_x += car_accel_x
        self.car_y += car_accel_y


        rotated_car = pygame.transform.rotate(self.car_image, self.car_angle)
        rotated_car_rect = rotated_car.get_rect()
        rotated_car_rect.center = (self.car_x + self.car_width // 2 - 0 * math.cos(math.radians(self.car_angle)),
                                self.car_y + self.car_height // 2 + 0 * math.sin(math.radians(self.car_angle)))

        


        

        rays = []
        for point in points_on_circle(Point(rotated_car_rect.centerx, rotated_car_rect.centery), 10000.0, 10, self.car_angle * 3.501):
            rays.append(HalfLine(Point(rotated_car_rect.centerx, rotated_car_rect.centery), point))

        points_of_intersection = []
        for i, ray in enumerate(rays):
            

            intersection_points = [ray.find_intersection(segment) for segment in self.borders]

            valid_points = [point for point in intersection_points if point is not None]
            if valid_points:
         
                points_of_intersection.append(min(valid_points, key=lambda p: (ray.origin.x - p.x)**2 + (ray.origin.y - p.y)**2))
               
            else:
                points_of_intersection.append(None)
        
        distances = []
        for point in  points_of_intersection:
            if point is not None:
                distances.append(Point(rotated_car_rect.centerx, rotated_car_rect.centery).distance_to(point))
            else:
                distances.append(2203.0)

        points = []
        for point in [Point(self.car_rect.topleft[0] + self.car_x , self.car_rect.topleft[1] + self.car_y ), Point(self.car_rect.topright[0] + self.car_x, self.car_rect.topright[1] + self.car_y), Point(self.car_rect.bottomleft[0] + self.car_x, self.car_rect.bottomleft[1] + self.car_y), Point(self.car_rect.bottomright[0] + self.car_x, self.car_rect.bottomright[1] + self.car_y)]:
            points.append(point)

        point_topleft, point_topright, point_bottomleft, point_bottomright = rotate_points(*points, -self.car_angle, rotated_car_rect.center)
        car_bounding_box = [Line_Segment(point_bottomleft, point_bottomright), Line_Segment(point_topleft, point_topright), Line_Segment(point_bottomleft, point_topleft), Line_Segment(point_bottomright, point_topright)]
        colliding = False
        got_reward = False
        for line_bounding_box in car_bounding_box:
            for line in self.borders:
                colliding = colliding or line_bounding_box.doIntersect(line)

        for line_bounding_box in car_bounding_box:
            for goal in self.goal_lines:
                got_reward = got_reward or (line_bounding_box.doIntersect(goal) and self.car_speed > 0 and not goal.has_collided_with)
                if line_bounding_box.doIntersect(goal) and self.car_speed > 0 and not goal.has_collided_with:
                    goal.has_collided_with = True
        
        if colliding:
            reward -= 100
            done = True
            self.reset()
            for goal in self.goal_lines:
                goal.has_collided_with = False
        
        if got_reward:
            reward += 200
        



        self.text = self.font.render(self.policy_loss, True, (0, 255, 0), (0, 0, 255))
        self.textRect = self.text.get_rect()
        self.textRect.centerx = self.screen_width // 2
        self.textRect.centery =self.screen_height // 2 -120
        

        next_state = np.concatenate(([self.car_angle % 360 / 360], [self.car_speed / self.max_speed], distances / np.array(2203)))
        next_state = np.array(next_state, dtype=np.float32)
        return next_state, reward, done, {}